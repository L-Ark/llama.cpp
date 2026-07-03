
#ifndef GGML_CUDA_MOE_STREAM_BATCH
#include <cstdint>
#include <cstddef>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
typedef struct { float direct_up; float direct_gate; float direct_fused; float repack_up; float repack_gate; float repack_fused; } ggml_moe_iq2_replay_result;
void ggml_cuda_moe_stream_batch_link_anchor(void) {}
void ggml_cuda_moe_ttft_trace_mark(const char *) {}
bool ggml_cuda_moe_iq2_prompt_replay(int, const void *, const void *, int64_t, int64_t, size_t, const float *, int64_t, int, float, ggml_moe_iq2_replay_result *) { return false; }
bool ggml_cuda_moe_stream_batch(int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t, const float *, size_t, size_t, float *, size_t, size_t, const int64_t *, const ggml_moe_row_mapping *, int64_t) { return false; }
bool ggml_cuda_moe_stream_preload_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_preload_tensor_prompt(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_register_tensor(int, const char *, const void *, int64_t, size_t, size_t) { return false; }
bool ggml_cuda_moe_stream_cache_contains(const char *, size_t, int) { return false; }
bool ggml_cuda_moe_stream_up_gate_batch(int, int, const char *, const void *, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, const float *, size_t, size_t, float *, size_t, size_t, int, float, const int64_t *, const ggml_moe_row_mapping *, int64_t) { return false; }
const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *, int, size_t) { return nullptr; }
}
#else
// Decode-only batched streaming MoE path.
//
// This file intentionally includes the MMQ-id kernel family without the MMVQ
// headers used by moe_stream.cu; several CUDA helper headers define unguarded
// device functions and cannot be mixed in one translation unit.

#include "common.cuh"
#include "ggml-quants.h"
#include "quantize.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
#if defined(__linux__) && !defined(GGML_MOE_DISABLE_LIBURING) && __has_include(<liburing.h>)
#include <liburing.h>
#define GGML_MOE_HAS_LIBURING 1
static inline void ggml_moe_io_uring_sqe_set_data64(struct io_uring_sqe * sqe, uint64_t data) {
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(static_cast<uintptr_t>(data)));
}
static inline uint64_t ggml_moe_io_uring_cqe_get_data64(const struct io_uring_cqe * cqe) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
}
#define io_uring_sqe_set_data64 ggml_moe_io_uring_sqe_set_data64
#define io_uring_cqe_get_data64 ggml_moe_io_uring_cqe_get_data64
#endif
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
    int  src0_up_type_int,
    int  src0_gate_type_int,
    const char *src0_up_name,
    const void *src0_up_data,
    const char *src0_gate_name,
    const void *src0_gate_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t up_nb01,
    size_t up_nb02,
    size_t up_expert_bytes,
    size_t gate_nb01,
    size_t gate_nb02,
    size_t gate_expert_bytes,
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
#if defined(GGML_MOE_HAS_LIBURING) && !defined(_WIN32)
    io_uring *uring = nullptr;
    size_t uring_depth = 0;
#endif
    uint64_t copies = 0;
    uint64_t waits = 0;
    uint64_t fallbacks = 0;
    uint64_t h2d_timed = 0;
    double slot_wait_ms = 0.0;
    double host_stage_ms = 0.0;
    double enqueue_ms = 0.0;
    double h2d_ms = 0.0;
    uint64_t iouring_batches = 0;
    uint64_t iouring_jobs = 0;
    uint64_t iouring_submit_calls = 0;
    uint64_t iouring_wait_calls = 0;
    uint64_t iouring_cqes = 0;
    uint64_t iouring_inflight_sum = 0;
    uint64_t iouring_inflight_samples = 0;
    uint64_t iouring_inflight_max = 0;
    uint64_t iouring_batch_hist[6] = {};
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

static size_t moe_stream_q8_1_row_bytes(int64_t ne00) {
    const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    return (size_t)ne00_padded * sizeof(block_q8_1) / QK8_1;
}

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
    uint64_t max_up_stage_jobs = 0;
    uint64_t max_gate_stage_jobs = 0;
    uint64_t max_combined_stage_jobs = 0;
    uint64_t up_stage_jobs_hist[7] = {};
    uint64_t gate_stage_jobs_hist[7] = {};
    uint64_t combined_stage_jobs_hist[7] = {};
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
    double wall_ms = 0.0;
};

static batch_profile g_bprof;
static batch_profile g_uprof;
static batch_profile g_uprof_by_type[2][GGML_TYPE_COUNT][GGML_TYPE_COUNT];

static size_t stage_jobs_hist_bucket(uint64_t jobs) {
    if (jobs == 0) return 0;
    if (jobs == 1) return 1;
    if (jobs <= 4) return 2;
    if (jobs <= 8) return 3;
    if (jobs <= 16) return 4;
    if (jobs <= 32) return 5;
    return 6;
}

struct up_gate_layer_profile_entry {
    bool used = false;
    bool prompt_mode = false;
    ggml_type up_type = GGML_TYPE_COUNT;
    ggml_type gate_type = GGML_TYPE_COUNT;
    char up_tensor[128] = {};
    char gate_tensor[128] = {};
    batch_profile profile;
};

static bool g_up_gate_layer_profile_enabled = false;
static std::atomic<uint64_t> g_up_gate_layer_profile_records{0};
static std::mutex g_up_gate_layer_profile_mu;
static up_gate_layer_profile_entry g_up_gate_layer_profiles[256];

static void up_gate_profile_add(
        batch_profile &p,
        uint64_t active_experts,
        uint64_t up_stage_jobs,
        uint64_t gate_stage_jobs,
        double stage_ms,
        double quant_ms,
        double up_ms,
        double gate_ms,
        double up_wait_ms,
        double gate_wait_ms,
        double up_compute_ms,
        double gate_compute_ms,
        double fuse_ms,
        double kernel_ms,
        double d2h_ms,
        double scatter_ms,
        double wall_ms) {
    ++p.calls;
    p.active_experts += active_experts;
    p.up_stage_jobs += up_stage_jobs;
    p.gate_stage_jobs += gate_stage_jobs;
    const uint64_t combined_stage_jobs = up_stage_jobs + gate_stage_jobs;
    if (p.max_up_stage_jobs < up_stage_jobs) {
        p.max_up_stage_jobs = up_stage_jobs;
    }
    if (p.max_gate_stage_jobs < gate_stage_jobs) {
        p.max_gate_stage_jobs = gate_stage_jobs;
    }
    if (p.max_combined_stage_jobs < combined_stage_jobs) {
        p.max_combined_stage_jobs = combined_stage_jobs;
    }
    ++p.up_stage_jobs_hist[stage_jobs_hist_bucket(up_stage_jobs)];
    ++p.gate_stage_jobs_hist[stage_jobs_hist_bucket(gate_stage_jobs)];
    ++p.combined_stage_jobs_hist[stage_jobs_hist_bucket(combined_stage_jobs)];
    p.stage_ms += stage_ms;
    p.quant_ms += quant_ms;
    p.up_ms += up_ms;
    p.gate_ms += gate_ms;
    p.up_wait_ms += up_wait_ms;
    p.gate_wait_ms += gate_wait_ms;
    p.up_compute_ms += up_compute_ms;
    p.gate_compute_ms += gate_compute_ms;
    p.fuse_ms += fuse_ms;
    p.kernel_ms += kernel_ms;
    p.d2h_ms += d2h_ms;
    p.scatter_ms += scatter_ms;
    p.wall_ms += wall_ms;
}

static void up_gate_type_profile_add(
        bool prompt_mode,
        ggml_type up_type,
        ggml_type gate_type,
        uint64_t active_experts,
        uint64_t up_stage_jobs,
        uint64_t gate_stage_jobs,
        double stage_ms,
        double quant_ms,
        double up_ms,
        double gate_ms,
        double up_wait_ms,
        double gate_wait_ms,
        double up_compute_ms,
        double gate_compute_ms,
        double fuse_ms,
        double kernel_ms,
        double d2h_ms,
        double scatter_ms,
        double wall_ms) {
    if (up_type < 0 || up_type >= GGML_TYPE_COUNT || gate_type < 0 || gate_type >= GGML_TYPE_COUNT) {
        return;
    }
    batch_profile &p = g_uprof_by_type[prompt_mode ? 1 : 0][up_type][gate_type];
    up_gate_profile_add(
        p,
        active_experts,
        up_stage_jobs,
        gate_stage_jobs,
        stage_ms,
        quant_ms,
        up_ms,
        gate_ms,
        up_wait_ms,
        gate_wait_ms,
        up_compute_ms,
        gate_compute_ms,
        fuse_ms,
        kernel_ms,
        d2h_ms,
        scatter_ms,
        wall_ms);
}

static void up_gate_layer_profile_add(
        bool prompt_mode,
        const char *up_tensor,
        const char *gate_tensor,
        ggml_type up_type,
        ggml_type gate_type,
        uint64_t active_experts,
        uint64_t up_stage_jobs,
        uint64_t gate_stage_jobs,
        double stage_ms,
        double quant_ms,
        double up_ms,
        double gate_ms,
        double up_wait_ms,
        double gate_wait_ms,
        double up_compute_ms,
        double gate_compute_ms,
        double fuse_ms,
        double kernel_ms,
        double d2h_ms,
        double scatter_ms,
        double wall_ms) {
    if (!g_up_gate_layer_profile_enabled || !up_tensor || !gate_tensor) {
        return;
    }
    ++g_up_gate_layer_profile_records;

    std::lock_guard<std::mutex> lk(g_up_gate_layer_profile_mu);
    up_gate_layer_profile_entry *free_entry = nullptr;
    up_gate_layer_profile_entry *entry = nullptr;
    for (up_gate_layer_profile_entry &candidate : g_up_gate_layer_profiles) {
        if (!candidate.used) {
            if (!free_entry) {
                free_entry = &candidate;
            }
            continue;
        }
        if (candidate.prompt_mode == prompt_mode &&
                candidate.up_type == up_type &&
                candidate.gate_type == gate_type &&
                std::strcmp(candidate.up_tensor, up_tensor) == 0 &&
                std::strcmp(candidate.gate_tensor, gate_tensor) == 0) {
            entry = &candidate;
            break;
        }
    }

    if (!entry) {
        entry = free_entry;
        if (!entry) {
            return;
        }
        entry->used = true;
        entry->prompt_mode = prompt_mode;
        entry->up_type = up_type;
        entry->gate_type = gate_type;
        std::snprintf(entry->up_tensor, sizeof(entry->up_tensor), "%s", up_tensor);
        std::snprintf(entry->gate_tensor, sizeof(entry->gate_tensor), "%s", gate_tensor);
        entry->profile = {};
    }

    up_gate_profile_add(
        entry->profile,
        active_experts,
        up_stage_jobs,
        gate_stage_jobs,
        stage_ms,
        quant_ms,
        up_ms,
        gate_ms,
        up_wait_ms,
        gate_wait_ms,
        up_compute_ms,
        gate_compute_ms,
        fuse_ms,
        kernel_ms,
        d2h_ms,
        scatter_ms,
        wall_ms);
}

struct current_down_overlap_profile {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> planned_jobs{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> missing_tensor{0};
    std::atomic<uint64_t> missing_pack{0};
    std::atomic<uint64_t> submitted_batches{0};
    std::atomic<uint64_t> completed_jobs{0};
    std::atomic<uint64_t> failed_batches{0};
    std::atomic<uint64_t> mark_failed{0};
    std::atomic<uint64_t> max_jobs{0};
    std::atomic<uint64_t> batch_hist_1{0};
    std::atomic<uint64_t> batch_hist_2_4{0};
    std::atomic<uint64_t> batch_hist_5_8{0};
    std::atomic<uint64_t> batch_hist_9_16{0};
    std::atomic<uint64_t> batch_hist_17_32{0};
    std::atomic<uint64_t> batch_hist_gt32{0};
    std::atomic<uint64_t> worker_us{0};
};

static current_down_overlap_profile g_current_down_overlap;

struct current_down_overlap_tensor_profile {
    uint64_t calls = 0;
    uint64_t planned_jobs = 0;
    uint64_t cache_hits = 0;
    uint64_t missing_tensor = 0;
    uint64_t missing_pack = 0;
    uint64_t completed_jobs = 0;
};

static std::mutex g_current_down_overlap_tensor_profile_mu;
static std::unordered_map<std::string, current_down_overlap_tensor_profile> g_current_down_overlap_tensor_profile;

struct expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    int32_t source_idx = 0;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct expert_pack_source {
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    char path[512] = {};
};

struct expert_pack_state {
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    std::vector<expert_pack_source> sources;
    std::vector<expert_pack_entry> entries;
    void *mmap_base = nullptr;
    size_t mmap_size = 0;
    bool mmap_attempted = false;
    bool mmap_enabled = false;
    std::atomic<uint64_t> mmap_hits{0};
    std::atomic<uint64_t> mmap_misses{0};
    std::atomic<uint64_t> mmap_bytes{0};
    bool inited = false;
    bool enabled = false;
    bool reported_io_backend = false;
    int io_backend = 0; // 0=buffered, 1=direct, 2=io_uring
    std::mutex mu;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_fallbacks{0};
    std::atomic<uint64_t> iouring_reads{0};
    std::atomic<uint64_t> iouring_bytes{0};
    std::atomic<uint64_t> iouring_fallbacks{0};
    std::atomic<uint64_t> iouring_submit_us{0};
    std::atomic<uint64_t> iouring_wait_us{0};
    std::atomic<uint64_t> iouring_h2d_enqueues{0};
    std::atomic<uint64_t> iouring_batches{0};
    std::atomic<uint64_t> iouring_submit_calls{0};
    std::atomic<uint64_t> iouring_wait_calls{0};
    std::atomic<uint64_t> iouring_cqes{0};
    std::atomic<uint64_t> iouring_inflight_sum{0};
    std::atomic<uint64_t> iouring_inflight_samples{0};
    std::atomic<uint64_t> iouring_inflight_max{0};
    std::atomic<uint64_t> iouring_batch_hist_1{0};
    std::atomic<uint64_t> iouring_batch_hist_2_4{0};
    std::atomic<uint64_t> iouring_batch_hist_5_8{0};
    std::atomic<uint64_t> iouring_batch_hist_9_16{0};
    std::atomic<uint64_t> iouring_batch_hist_17_32{0};
    std::atomic<uint64_t> iouring_batch_hist_gt32{0};
    // RAM hot tier
    void *ram_tier_base = nullptr;
    size_t ram_tier_bytes = 0;
    struct ram_tier_entry { size_t offset; size_t nbytes; };
    std::vector<ram_tier_entry> ram_tier_index; // parallel to entries[], non-zero offset = resident
    size_t ram_tier_pinned_bytes = 0;
    std::atomic<uint64_t> ram_tier_hits{0};
    std::atomic<uint64_t> ram_tier_total{0};
};

static expert_pack_state g_expert_pack;

static void batch_profile_report_atexit() {
    if (!g_bprof.enabled || g_bprof.calls == 0) return;
    const double calls = (double)g_bprof.calls;
    const double accounted_ms =
        g_bprof.stage_ms + g_bprof.quant_ms + g_bprof.kernel_ms + g_bprof.d2h_ms + g_bprof.scatter_ms;
    const double wall_gap_ms = g_bprof.wall_ms - accounted_ms;
    std::fprintf(stderr,
        "[moe_stream_batch] profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
        "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
        g_bprof.calls,
        (double)g_bprof.active_experts / calls,
        g_bprof.stage_ms / calls,
        g_bprof.quant_ms / calls,
        g_bprof.kernel_ms / calls,
        g_bprof.d2h_ms / calls,
        g_bprof.scatter_ms / calls,
        accounted_ms / calls,
        g_bprof.wall_ms / calls,
        wall_gap_ms / calls);
}

static void up_gate_profile_report_atexit() {
    if (!g_uprof.enabled || g_uprof.calls == 0) return;
    const double calls = (double)g_uprof.calls;
    const double accounted_ms =
        g_uprof.stage_ms + g_uprof.quant_ms + g_uprof.kernel_ms + g_uprof.d2h_ms + g_uprof.scatter_ms;
    const double wall_gap_ms = g_uprof.wall_ms - accounted_ms;
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms up=%.3f ms gate=%.3f ms "
        "up_stage_jobs=%.2f gate_stage_jobs=%.2f "
        "up_wait=%.3f ms gate_wait=%.3f ms up_compute=%.3f ms gate_compute=%.3f ms fuse=%.3f ms "
        "kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
        "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
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
        accounted_ms / calls,
        g_uprof.wall_ms / calls,
        wall_gap_ms / calls);
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate stage-job hist: "
        "max_up=%lu max_gate=%lu max_combined=%lu "
        "up=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
        "gate=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
        "combined=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
        g_uprof.max_up_stage_jobs,
        g_uprof.max_gate_stage_jobs,
        g_uprof.max_combined_stage_jobs,
        g_uprof.up_stage_jobs_hist[0], g_uprof.up_stage_jobs_hist[1],
        g_uprof.up_stage_jobs_hist[2], g_uprof.up_stage_jobs_hist[3],
        g_uprof.up_stage_jobs_hist[4], g_uprof.up_stage_jobs_hist[5],
        g_uprof.up_stage_jobs_hist[6],
        g_uprof.gate_stage_jobs_hist[0], g_uprof.gate_stage_jobs_hist[1],
        g_uprof.gate_stage_jobs_hist[2], g_uprof.gate_stage_jobs_hist[3],
        g_uprof.gate_stage_jobs_hist[4], g_uprof.gate_stage_jobs_hist[5],
        g_uprof.gate_stage_jobs_hist[6],
        g_uprof.combined_stage_jobs_hist[0], g_uprof.combined_stage_jobs_hist[1],
        g_uprof.combined_stage_jobs_hist[2], g_uprof.combined_stage_jobs_hist[3],
        g_uprof.combined_stage_jobs_hist[4], g_uprof.combined_stage_jobs_hist[5],
        g_uprof.combined_stage_jobs_hist[6]);
    for (int mode = 0; mode < 2; ++mode) {
        for (int up_type = 0; up_type < GGML_TYPE_COUNT; ++up_type) {
            for (int gate_type = 0; gate_type < GGML_TYPE_COUNT; ++gate_type) {
                const batch_profile &p = g_uprof_by_type[mode][up_type][gate_type];
                if (p.calls == 0) {
                    continue;
                }
                const double pcalls = (double)p.calls;
                const double p_accounted_ms =
                    p.stage_ms + p.quant_ms + p.kernel_ms + p.d2h_ms + p.scatter_ms;
                const double p_wall_gap_ms = p.wall_ms - p_accounted_ms;
                std::fprintf(stderr,
                    "[moe_stream_batch] up/gate type profile: mode=%s up_type=%d gate_type=%d "
                    "calls=%lu avg_active=%.2f stage=%.3f ms quant=%.3f ms "
                    "up=%.3f ms gate=%.3f ms up_stage_jobs=%.2f gate_stage_jobs=%.2f "
                    "up_wait=%.3f ms gate_wait=%.3f ms up_compute=%.3f ms gate_compute=%.3f ms "
                    "fuse=%.3f ms kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms "
                    "total=%.3f ms/call wall=%.3f ms/call wall_gap=%.3f ms/call\n",
                    mode ? "prompt" : "decode",
                    up_type,
                    gate_type,
                    p.calls,
                    (double)p.active_experts / pcalls,
                    p.stage_ms / pcalls,
                    p.quant_ms / pcalls,
                    p.up_ms / pcalls,
                    p.gate_ms / pcalls,
                    (double)p.up_stage_jobs / pcalls,
                    (double)p.gate_stage_jobs / pcalls,
                    p.up_wait_ms / pcalls,
                    p.gate_wait_ms / pcalls,
                    p.up_compute_ms / pcalls,
                    p.gate_compute_ms / pcalls,
                    p.fuse_ms / pcalls,
                    p.kernel_ms / pcalls,
                    p.d2h_ms / pcalls,
                    p.scatter_ms / pcalls,
                    p_accounted_ms / pcalls,
                    p.wall_ms / pcalls,
                    p_wall_gap_ms / pcalls);
                std::fprintf(stderr,
                    "[moe_stream_batch] up/gate type stage-job hist: mode=%s up_type=%d gate_type=%d "
                    "max_up=%lu max_gate=%lu max_combined=%lu "
                    "up=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
                    "gate=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu "
                    "combined=0:%lu,1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
                    mode ? "prompt" : "decode",
                    up_type,
                    gate_type,
                    p.max_up_stage_jobs,
                    p.max_gate_stage_jobs,
                    p.max_combined_stage_jobs,
                    p.up_stage_jobs_hist[0], p.up_stage_jobs_hist[1],
                    p.up_stage_jobs_hist[2], p.up_stage_jobs_hist[3],
                    p.up_stage_jobs_hist[4], p.up_stage_jobs_hist[5],
                    p.up_stage_jobs_hist[6],
                    p.gate_stage_jobs_hist[0], p.gate_stage_jobs_hist[1],
                    p.gate_stage_jobs_hist[2], p.gate_stage_jobs_hist[3],
                    p.gate_stage_jobs_hist[4], p.gate_stage_jobs_hist[5],
                    p.gate_stage_jobs_hist[6],
                    p.combined_stage_jobs_hist[0], p.combined_stage_jobs_hist[1],
                    p.combined_stage_jobs_hist[2], p.combined_stage_jobs_hist[3],
                    p.combined_stage_jobs_hist[4], p.combined_stage_jobs_hist[5],
                    p.combined_stage_jobs_hist[6]);
            }
        }
    }
}

static void up_gate_layer_profile_report_atexit() {
    if (!g_up_gate_layer_profile_enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate layer profile summary: records=%lu\n",
        (unsigned long)g_up_gate_layer_profile_records.load(std::memory_order_relaxed));

    up_gate_layer_profile_entry entries[256];
    size_t n_entries = 0;
    {
        std::lock_guard<std::mutex> lk(g_up_gate_layer_profile_mu);
        for (const up_gate_layer_profile_entry &entry : g_up_gate_layer_profiles) {
            if (entry.used && entry.profile.calls > 0 && n_entries < 256) {
                entries[n_entries++] = entry;
            }
        }
    }

    if (n_entries == 0) {
        std::fprintf(stderr, "[moe_stream_batch] up/gate layer profile: no entries recorded\n");
        return;
    }

    std::sort(entries, entries + n_entries,
        [](const up_gate_layer_profile_entry &a, const up_gate_layer_profile_entry &b) {
            return a.profile.wall_ms > b.profile.wall_ms;
        });

    const char *limit_env = std::getenv("GGML_MOE_UP_GATE_LAYER_PROFILE_TOP");
    int limit = 32;
    if (limit_env && limit_env[0]) {
        limit = std::atoi(limit_env);
        if (limit <= 0) {
            limit = 32;
        }
    }

    int printed = 0;
    for (size_t i = 0; i < n_entries && printed < limit; ++i) {
        const up_gate_layer_profile_entry &entry = entries[i];
        const batch_profile &p = entry.profile;
        const double calls = (double)p.calls;
        const double accounted_ms = p.stage_ms + p.quant_ms + p.kernel_ms + p.d2h_ms + p.scatter_ms;
        const double wall_gap_ms = p.wall_ms - accounted_ms;
        std::fprintf(stderr,
            "[moe_stream_batch] up/gate layer profile: rank=%d mode=%s up_tensor=%s gate_tensor=%s "
            "up_type=%d gate_type=%d calls=%lu active=%lu avg_active=%.2f "
            "stage_total=%.3f quant_total=%.3f up_total=%.3f gate_total=%.3f "
            "up_wait_total=%.3f gate_wait_total=%.3f up_compute_total=%.3f gate_compute_total=%.3f "
            "fuse_total=%.3f kernel_total=%.3f d2h_total=%.3f scatter_total=%.3f wall_total=%.3f "
            "wall_per_call=%.3f kernel_per_call=%.3f wall_gap_total=%.3f "
            "up_stage_jobs=%lu gate_stage_jobs=%lu\n",
            printed + 1,
            entry.prompt_mode ? "prompt" : "decode",
            entry.up_tensor,
            entry.gate_tensor,
            (int)entry.up_type,
            (int)entry.gate_type,
            p.calls,
            p.active_experts,
            p.active_experts / calls,
            p.stage_ms,
            p.quant_ms,
            p.up_ms,
            p.gate_ms,
            p.up_wait_ms,
            p.gate_wait_ms,
            p.up_compute_ms,
            p.gate_compute_ms,
            p.fuse_ms,
            p.kernel_ms,
            p.d2h_ms,
            p.scatter_ms,
            p.wall_ms,
            p.wall_ms / calls,
            p.kernel_ms / calls,
            wall_gap_ms,
            p.up_stage_jobs,
            p.gate_stage_jobs);
        ++printed;
    }
}

struct batch_vram_cache {
    void * pool = nullptr;
    size_t slot_sz = 0;
    int n_slots = 0;
    uintptr_t slot_key[16384] = {};
    uint64_t slot_used[16384] = {};
    uint32_t slot_hits[16384] = {};
    uint32_t slot_profile_count[16384] = {};
    bool slot_pinned[16384] = {};
    bool slot_prefetch_down[16384] = {};
    bool slot_pending[16384] = {};
    cudaEvent_t slot_ready[16384] = {};
    uint64_t clock = 1;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    uint64_t pinned = 0;
    uint64_t down_prefetch_loads = 0;
    uint64_t down_prefetch_hits = 0;
    uint64_t down_prefetch_evicted = 0;
    uint64_t async_prefetch_waits = 0;
};

static constexpr int BATCH_VRAM_CACHE_COUNT = 3;
static constexpr int BATCH_VRAM_CACHE_UPGATE = 1;
static constexpr int BATCH_VRAM_CACHE_Q40_DOWN = 2;

static batch_vram_cache g_bcaches[BATCH_VRAM_CACHE_COUNT];
static bool g_bcache_inited[BATCH_VRAM_CACHE_COUNT] = {};

static const char * batch_cache_label(int cid) {
    switch (cid) {
        case BATCH_VRAM_CACHE_UPGATE: return "upgate";
        case BATCH_VRAM_CACHE_Q40_DOWN: return "q4_0_down";
        default: return "down";
    }
}

struct cache_policy_diag {
    std::atomic<uint64_t> profile_count_lookups{0};
    std::atomic<uint64_t> profile_count_hits{0};
    std::atomic<uint64_t> inserted_profile_count_sum{0};
    std::atomic<uint64_t> inserted_profile_count_nonzero{0};
    std::atomic<uint64_t> profile_policy_evictions{0};
    std::atomic<uint64_t> profile_policy_victim_count_sum{0};
    std::atomic<uint64_t> profile_policy_victim_count_nonzero{0};
};

static cache_policy_diag g_cache_policy_diag;
static std::atomic<bool> g_cache_policy_diag_registered{false};

struct profile_entry {
    int expert_idx = -1;
    uint64_t count = 0;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

static std::vector<profile_entry> g_profile;
static std::unordered_map<uint64_t, uint32_t> g_profile_counts;
static std::unordered_set<uint64_t> g_profile_pinned_keys;
static bool g_profile_loaded = false;
static bool g_profile_enabled = false;
static std::vector<profile_entry> g_prompt_profile;
static bool g_prompt_profile_loaded = false;
static bool g_prompt_profile_enabled = false;
static std::mutex g_profile_mu;
static std::mutex g_profile_pinned_mu;
static char g_preloaded_tensors[256][128] = {};
static int g_n_preloaded_tensors = 0;
static std::atomic<int> g_profile_preload_calls{0};

struct batch_route_profile_entry {
    int expert_idx = -1;
    uint64_t count = 0;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

struct batch_route_trace_entry {
    uint64_t seq = 0;
    int expert_idx = -1;
    size_t expert_bytes = 0;
    char tensor[128] = {};
};

struct trace_prefetch_state {
    bool inited = false;
    bool enabled = false;
    bool reported = false;
    size_t cursor = 0;
    size_t prefetch_cursor = 0;
    size_t lead_events = 0;
    size_t window = 96;
    int max_loads = 8;
    std::vector<batch_route_trace_entry> trace;
    std::mutex mu;
    uint64_t calls = 0;
    uint64_t matched = 0;
    uint64_t resync = 0;
    uint64_t loads = 0;
    uint64_t cached = 0;
    uint64_t missing_tensor = 0;
    uint64_t cache_unavailable = 0;
};

struct host_prefetch_slot {
    void *host = nullptr;
    cudaEvent_t done = nullptr;
    size_t capacity = 0;
    const expert_pack_entry *entry = nullptr;
    size_t nbytes = 0;
    char tensor[128] = {};
    int expert_idx = -1;
    bool ready = false;
    bool reserved = false;
    bool in_use = false;
};

struct host_prefetch_state {
    bool inited = false;
    bool enabled = false;
    bool planned_enabled = false;
    bool stop = false;
    bool failed = false;
    size_t cursor = 0;
    size_t produce_cursor = 0;
    size_t skip_events = 0;
    size_t lead_events = 2048;
    size_t max_bytes = 512ULL * 1024ULL * 1024ULL;
    size_t used_bytes = 0;
    std::vector<batch_route_trace_entry> trace;
    std::deque<batch_route_trace_entry> planned;
    std::vector<host_prefetch_slot> slots;
    std::unordered_map<uint64_t, size_t> ready;
    std::unordered_map<uint64_t, size_t> planned_queued;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    uint64_t calls = 0;
    uint64_t matched = 0;
    uint64_t resync = 0;
    uint64_t submitted = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evicted = 0;
    uint64_t read_failures = 0;
    uint64_t alloc_failures = 0;
    uint64_t duplicate_skips = 0;
    uint64_t profile_pinned_skips = 0;
    uint64_t no_slot = 0;
    uint64_t scan_passes = 0;
    uint64_t reserved_skips = 0;
    uint64_t planned_enqueued = 0;
    uint64_t planned_dequeued = 0;
    uint64_t planned_duplicate_skips = 0;
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
    char tensor[128] = {};
};

static std::vector<batch_route_profile_entry> g_route_profile;
static std::vector<batch_route_trace_entry> g_route_trace;
static trace_prefetch_state g_trace_prefetch;
static host_prefetch_state g_host_prefetch;
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

static int trace_prefetch_env_int(const char *name, int fallback, int lo, int hi) {
    const char *env = std::getenv(name);
    long value = (env && env[0]) ? std::atol(env) : fallback;
    if (value < lo) value = lo;
    if (value > hi) value = hi;
    return (int)value;
}

static void trim_csv_field_tail(char *s) {
    if (!s) return;
    size_t n = std::strlen(s);
    while (n > 0 && (s[n - 1] == '\r' || s[n - 1] == '\n')) {
        s[--n] = '\0';
    }
}

static bool load_route_trace_csv(const char *path, std::vector<batch_route_trace_entry> &trace, const char *label) {
    if (!path || !path[0]) return false;
    FILE *f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] %s: open failed: %s\n", label, path);
        return false;
    }
    char line[512];
    if (std::fgets(line, sizeof(line), f)) {
        while (std::fgets(line, sizeof(line), f)) {
            batch_route_trace_entry e;
            unsigned long long seq = 0;
            unsigned long long tensor_base = 0;
            if (std::sscanf(line, "%llu,%zu,0x%llx,%d,%127[^\n]",
                        &seq, &e.expert_bytes, &tensor_base, &e.expert_idx, e.tensor) == 5 &&
                    e.expert_idx >= 0 && e.expert_bytes > 0 && e.tensor[0]) {
                trim_csv_field_tail(e.tensor);
                e.seq = (uint64_t)seq;
                trace.push_back(e);
            }
        }
    }
    std::fclose(f);
    return !trace.empty();
}

static void trace_prefetch_report_atexit() {
    std::lock_guard<std::mutex> lk(g_trace_prefetch.mu);
    if (!g_trace_prefetch.enabled || g_trace_prefetch.calls == 0) return;
    std::fprintf(stderr,
        "[moe_stream_batch] trace prefetch: calls=%lu matched=%lu resync=%lu loads=%lu cached=%lu missing_tensor=%lu cache_unavailable=%lu cursor=%zu prefetch_cursor=%zu/%zu\n",
        g_trace_prefetch.calls,
        g_trace_prefetch.matched,
        g_trace_prefetch.resync,
        g_trace_prefetch.loads,
        g_trace_prefetch.cached,
        g_trace_prefetch.missing_tensor,
        g_trace_prefetch.cache_unavailable,
        g_trace_prefetch.cursor,
        g_trace_prefetch.prefetch_cursor,
        g_trace_prefetch.trace.size());
}

static void trace_prefetch_init_once() {
    if (g_trace_prefetch.inited) return;
    std::lock_guard<std::mutex> lk(g_trace_prefetch.mu);
    if (g_trace_prefetch.inited) return;

    const char *path = std::getenv("GGML_MOE_TRACE_PREFETCH");
    if (!path || !path[0]) {
        g_trace_prefetch.inited = true;
        return;
    }

    load_route_trace_csv(path, g_trace_prefetch.trace, "trace prefetch");

    g_trace_prefetch.window = (size_t)trace_prefetch_env_int("GGML_MOE_TRACE_PREFETCH_WINDOW", 96, 1, 4096);
    g_trace_prefetch.max_loads = trace_prefetch_env_int("GGML_MOE_TRACE_PREFETCH_MAX_LOADS", 8, 1, 256);
    g_trace_prefetch.lead_events = (size_t)trace_prefetch_env_int("GGML_MOE_TRACE_PREFETCH_LEAD_EVENTS", 0, 0, 4096);
    g_trace_prefetch.enabled = !g_trace_prefetch.trace.empty();
    g_trace_prefetch.inited = true;
    if (g_trace_prefetch.enabled) {
        std::atexit(trace_prefetch_report_atexit);
        std::fprintf(stderr,
            "[moe_stream_batch] trace prefetch: loaded %zu events from %s window=%zu max_loads=%d lead_events=%zu\n",
            g_trace_prefetch.trace.size(), path, g_trace_prefetch.window, g_trace_prefetch.max_loads,
            g_trace_prefetch.lead_events);
    }
}

static void trace_prefetch_on_hit(const char *tensor_name, int expert_idx, size_t expert_bytes);
static void host_prefetch_on_route(const char *tensor_name, int expert_idx, size_t expert_bytes);

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
    const bool is_marker = op && std::strncmp(op, "mark_", 5) == 0;
    if (!is_marker && g_ttft_trace.size() >= g_ttft_trace_cap) return;

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
    trace_prefetch_on_hit(tensor_name, expert_idx, expert_bytes);
    host_prefetch_on_route(tensor_name, expert_idx, expert_bytes);
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
    uint64_t async_prefetch_waits = 0;
    for (batch_vram_cache &c : g_bcaches) {
        hits += c.hits;
        misses += c.misses;
        preloads += c.preloads;
        pinned += c.pinned;
        down_prefetch_loads += c.down_prefetch_loads;
        down_prefetch_hits += c.down_prefetch_hits;
        down_prefetch_evicted += c.down_prefetch_evicted;
        async_prefetch_waits += c.async_prefetch_waits;
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
    for (int ic = 0; ic < BATCH_VRAM_CACHE_COUNT; ++ic) {
        const batch_vram_cache &c = g_bcaches[ic];
        const uint64_t c_total = c.hits + c.misses;
        if (c_total == 0) continue;
        const char *label = batch_cache_label(ic);
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
    if (async_prefetch_waits > 0) {
        std::fprintf(stderr, "[moe_stream_batch] async prefetch waits=%lu\n", async_prefetch_waits);
    }
}

static bool profile_protect_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PROTECT");
    return env && env[0] && env[0] != '0';
}

static void profile_pinned_key_record(uint64_t key) {
    if (key == 0) return;
    std::lock_guard<std::mutex> lk(g_profile_pinned_mu);
    g_profile_pinned_keys.insert(key);
}

static bool profile_pinned_key_contains(uint64_t key) {
    if (key == 0) return false;
    std::lock_guard<std::mutex> lk(g_profile_pinned_mu);
    return g_profile_pinned_keys.find(key) != g_profile_pinned_keys.end();
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
        if (std::sscanf(line, "%llu,%llu,%zu,%llu,0x%llx,%d,%127[^\n]",
                    &rank, &count, &expert_bytes, &cumulative, &tensor_base, &e.expert_idx, e.tensor) == 7 &&
                e.expert_idx >= 0 && e.tensor[0]) {
            trim_csv_field_tail(e.tensor);
            e.count = (uint64_t)count;
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

static void rebuild_profile_count_index_locked() {
    g_profile_counts.clear();
    for (const profile_entry &e : g_profile) {
        const uint64_t count = e.count > UINT32_MAX ? UINT32_MAX : e.count;
        g_profile_counts[batch_key_hash(e.tensor, e.expert_idx)] = (uint32_t)count;
    }
}

static uint32_t profile_count_for_key(uintptr_t key) {
    std::lock_guard<std::mutex> lk(g_profile_mu);
    ++g_cache_policy_diag.profile_count_lookups;
    const auto it = g_profile_counts.find((uint64_t)key);
    if (it == g_profile_counts.end()) return 0;
    ++g_cache_policy_diag.profile_count_hits;
    return it->second;
}

static void load_profile_once() {
    if (g_profile_loaded) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_profile_loaded) return;
    const char *path = std::getenv("GGML_MOE_VRAM_PROFILE");
    g_profile_enabled = load_profile_file(path, g_profile, "profile");
    if (g_profile_enabled) {
        rebuild_profile_count_index_locked();
    }
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
    const char *split_max_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB");
    size_t split_max_mib = 4;
    if (split_max_env && split_max_env[0]) {
        split_max_mib = (size_t)std::strtoull(split_max_env, nullptr, 10);
        if (split_max_mib < 1) split_max_mib = 1;
        if (split_max_mib > 64) split_max_mib = 64;
    }
    if (fused_env && fused_env[0] && fused_env[0] != '0' &&
            split_env && split_env[0] && split_env[0] != '0' &&
            expert_sz <= split_max_mib*1024ULL*1024ULL) {
        return 1;
    }
    if (fused_env && fused_env[0] && fused_env[0] != '0' && expert_sz <= 2ULL*1024ULL*1024ULL) {
        return 1;
    }
    return 0;
}

static size_t batch_cache_q40_budget_mib() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_Q40_MIB");
    if (!env || !env[0]) return 0;
    size_t mib = (size_t)std::strtoull(env, nullptr, 10);
    if (mib > 2048) mib = 2048;
    return mib;
}

static size_t batch_cache_budget_mib_for_id(size_t budget_mib, int cid) {
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    const bool fused = fused_env && fused_env[0] && fused_env[0] != '0';
    const char *split_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT");
    const bool split = fused && split_env && split_env[0] && split_env[0] != '0';
    const size_t q40_mib = batch_cache_q40_budget_mib();

    if (cid == BATCH_VRAM_CACHE_Q40_DOWN) {
        return q40_mib;
    }

    if (split) {
        const char *pct_env = std::getenv("GGML_MOE_VRAM_CACHE_UPGATE_PCT");
        long upgate_pct = (pct_env && pct_env[0]) ? std::atol(pct_env) : 63;
        if (upgate_pct < 1) upgate_pct = 1;
        if (upgate_pct > 99) upgate_pct = 99;
        const size_t upgate_mib = (budget_mib * (size_t)upgate_pct) / 100ULL;
        if (cid == BATCH_VRAM_CACHE_UPGATE) return upgate_mib > 0 ? upgate_mib : 1;
        size_t down_mib = budget_mib > upgate_mib ? budget_mib - upgate_mib : 1;
        if (q40_mib > 0 && down_mib > q40_mib + 1) down_mib -= q40_mib;
        return down_mib;
    }

    if (fused && budget_mib > 8192 && cid == BATCH_VRAM_CACHE_UPGATE) {
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

static batch_vram_cache * batch_cache_get_for_id(size_t expert_sz, int cid) {
    if (cid < 0 || cid >= BATCH_VRAM_CACHE_COUNT) return nullptr;
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
        std::fill_n(c->slot_profile_count, 16384, (uint32_t)0);
        std::fill_n(c->slot_pinned, 16384, false);
        std::fill_n(c->slot_prefetch_down, 16384, false);
        for (cudaEvent_t &ev : c->slot_ready) {
            if (ev) {
                cudaEventDestroy(ev);
                ev = nullptr;
            }
        }
        std::fill_n(c->slot_pending, 16384, false);
        c->clock = 1;
        c->hits = 0;
        c->misses = 0;
        c->preloads = 0;
        c->pinned = 0;
        c->down_prefetch_loads = 0;
        c->down_prefetch_hits = 0;
        c->down_prefetch_evicted = 0;
        c->async_prefetch_waits = 0;
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
    int alloc_slots = c->n_slots;
    size_t alloc = (size_t)alloc_slots * expert_sz;
    cudaError_t alloc_err = cudaMalloc(&c->pool, alloc);
    if (alloc_err != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc %.1f GiB FAILED; retrying smaller pool\n",
                     alloc / (1024.0*1024.0*1024.0));
        cudaGetLastError();
        const char *retry_env = std::getenv("GGML_MOE_VRAM_CACHE_ALLOC_RETRY");
        const bool retry = !retry_env || !retry_env[0] || retry_env[0] != '0';
        while (retry && alloc_slots > 1) {
            int next_slots = (alloc_slots * 7) / 8;
            if (next_slots >= alloc_slots) next_slots = alloc_slots - 1;
            alloc_slots = next_slots;
            alloc = (size_t)alloc_slots * expert_sz;
            alloc_err = cudaMalloc(&c->pool, alloc);
            if (alloc_err == cudaSuccess) break;
            cudaGetLastError();
        }
        if (alloc_err != cudaSuccess) {
            std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc retry failed; disabling %s cache\n",
                         batch_cache_label(cid));
            c->n_slots = 0;
            g_bcache_inited[cid] = true;
            return nullptr;
        }
        c->n_slots = alloc_slots;
    }
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache %s: %.1f GiB, %d slots (%.2f MiB each)\n",
                 batch_cache_label(cid), alloc / (1024.0*1024.0*1024.0),
                 c->n_slots, expert_sz / (1024.0*1024.0));
    std::atexit(batch_cache_report_atexit);
    g_bcache_inited[cid] = true;
    return c;
}

static batch_vram_cache * batch_cache_get(size_t expert_sz) {
    return batch_cache_get_for_id(expert_sz, batch_cache_id_for_size(expert_sz));
}

static bool batch_cache_wait_slot_ready(batch_vram_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->n_slots || !c->slot_pending[slot]) return true;
    if (c->slot_ready[slot]) {
        if (cudaEventSynchronize(c->slot_ready[slot]) != cudaSuccess) {
            return false;
        }
    }
    c->slot_pending[slot] = false;
    ++c->async_prefetch_waits;
    return true;
}

static int batch_cache_lookup_slot(batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return -1;
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) {
            if (!batch_cache_wait_slot_ready(c, slot)) return -1;
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
    batch_cache_wait_slot_ready(c, slot);
    c->slot_key[slot] = 0;
    c->slot_used[slot] = 0;
    c->slot_hits[slot] = 0;
    c->slot_profile_count[slot] = 0;
    if (c->slot_pinned[slot] && c->pinned > 0) {
        --c->pinned;
    }
    c->slot_pinned[slot] = false;
    c->slot_prefetch_down[slot] = false;
    c->slot_pending[slot] = false;
}

static int batch_cache_find_slot(batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return -1;
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) return slot;
    }
    return -1;
}

static bool batch_cache_promote_profile_slot(batch_vram_cache *c, int slot, uint64_t profile_count) {
    if (!c || slot < 0 || slot >= c->n_slots || c->slot_key[slot] == 0) return false;
    if (c->slot_pinned[slot]) {
        if (profile_count > 0) {
            c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
        }
        return true;
    }
    if (c->pinned >= profile_preload_slot_budget(c)) return false;
    c->slot_pinned[slot] = true;
    ++c->pinned;
    c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
    return true;
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

static bool cache_policy_profile_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "profile_lfu_lru") == 0;
}

static bool cache_policy_hybrid_profile_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "hybrid_profile_lfu_lru") == 0;
}

static uint64_t cache_policy_hybrid_after() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_PROFILE_AFTER");
    if (!env || !env[0]) return 12624;
    return std::strtoull(env, nullptr, 10);
}

static void cache_policy_diag_report_atexit() {
    const uint64_t lookups = g_cache_policy_diag.profile_count_lookups.load();
    const uint64_t evictions = g_cache_policy_diag.profile_policy_evictions.load();
    if (lookups == 0 && evictions == 0) return;
    const uint64_t lookup_hits = g_cache_policy_diag.profile_count_hits.load();
    const uint64_t inserted_nonzero = g_cache_policy_diag.inserted_profile_count_nonzero.load();
    const uint64_t inserted_sum = g_cache_policy_diag.inserted_profile_count_sum.load();
    const uint64_t victim_nonzero = g_cache_policy_diag.profile_policy_victim_count_nonzero.load();
    const uint64_t victim_sum = g_cache_policy_diag.profile_policy_victim_count_sum.load();
    std::fprintf(stderr,
        "[moe_stream_batch] cache policy diag: profile_count_lookups=%lu hits=%lu "
        "inserted_nonzero=%lu inserted_avg=%.2f evictions=%lu victim_nonzero=%lu victim_avg=%.2f\n",
        lookups, lookup_hits, inserted_nonzero,
        inserted_nonzero > 0 ? (double)inserted_sum / (double)inserted_nonzero : 0.0,
        evictions, victim_nonzero,
        victim_nonzero > 0 ? (double)victim_sum / (double)victim_nonzero : 0.0);
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

static size_t expert_pack_env_size(const char *name, size_t default_value, size_t min_value, size_t max_value) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    const unsigned long long parsed = std::strtoull(env, nullptr, 10);
    if (parsed == 0) return default_value;
    size_t value = (size_t)parsed;
    if (value < min_value) value = min_value;
    if (value > max_value) value = max_value;
    return value;
}

static bool expert_pack_env_bool(const char *name, bool default_value) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    return env[0] != '0';
}

static size_t expert_pack_io_bytes() {
    return expert_pack_env_size("GGML_MOE_IO_BYTES", 2ULL * 1024ULL * 1024ULL,
            expert_pack_direct_alignment(), 64ULL * 1024ULL * 1024ULL);
}

static size_t expert_pack_io_depth() {
    return expert_pack_env_size("GGML_MOE_IO_DEPTH", 32, 1, 256);
}

static size_t expert_pack_io_refill_batch() {
    return expert_pack_env_size("GGML_MOE_IO_REFILL_BATCH", 1, 1, 256);
}

static void expert_pack_atomic_max(std::atomic<uint64_t> &target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
            !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static size_t expert_pack_iouring_batch_bucket(size_t jobs) {
    if (jobs <= 1) return 0;
    if (jobs <= 4) return 1;
    if (jobs <= 8) return 2;
    if (jobs <= 16) return 3;
    if (jobs <= 32) return 4;
    return 5;
}

static void expert_pack_record_iouring_batch(size_t jobs) {
    ++g_expert_pack.iouring_batches;
    switch (expert_pack_iouring_batch_bucket(jobs)) {
        case 0: ++g_expert_pack.iouring_batch_hist_1; break;
        case 1: ++g_expert_pack.iouring_batch_hist_2_4; break;
        case 2: ++g_expert_pack.iouring_batch_hist_5_8; break;
        case 3: ++g_expert_pack.iouring_batch_hist_9_16; break;
        case 4: ++g_expert_pack.iouring_batch_hist_17_32; break;
        default: ++g_expert_pack.iouring_batch_hist_gt32; break;
    }
}

static bool down_batch_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_DOWN_BATCH_PROFILE_OUT");
    return env && env[0];
}

static void down_batch_profile_record(
        const char *tensor,
        ggml_type src0_type,
        int n_active,
        int cache_hits,
        int cache_misses,
        int staged_jobs,
        float stage_ms,
        float quant_ms,
        float kernel_ms,
        float d2h_ms,
        double scatter_ms,
        double wall_ms) {
    const char *path = std::getenv("GGML_MOE_DOWN_BATCH_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,tensor,src0_type,n_active,cache_hits,cache_misses,staged_jobs,stage_ms,quant_ms,kernel_ms,d2h_ms,scatter_ms,wall_ms\n");
        header_written = true;
    }

    static uint64_t seq = 0;
    std::fprintf(f,
            "%lu,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned long)++seq,
            tensor ? tensor : "",
            (int)src0_type,
            n_active,
            cache_hits,
            cache_misses,
            staged_jobs,
            (double)stage_ms,
            (double)quant_ms,
            (double)kernel_ms,
            (double)d2h_ms,
            scatter_ms,
            wall_ms);
    std::fclose(f);
}

static bool up_gate_profile_csv_enabled() {
    const char *env = std::getenv("GGML_MOE_UP_GATE_PROFILE_OUT");
    return env && env[0];
}

static void up_gate_profile_csv_record(
        bool prompt_mode,
        const char *up_tensor,
        const char *gate_tensor,
        ggml_type up_type,
        ggml_type gate_type,
        int n_active,
        int up_cache_hits,
        int up_cache_misses,
        int gate_cache_hits,
        int gate_cache_misses,
        int up_stage_jobs,
        int gate_stage_jobs,
        float stage_ms,
        float quant_ms,
        float up_ms,
        float gate_ms,
        float up_wait_ms,
        float gate_wait_ms,
        float up_compute_ms,
        float gate_compute_ms,
        float fuse_ms,
        float kernel_ms,
        float d2h_ms,
        double scatter_ms,
        double wall_ms,
        bool use_handoff,
        bool parallel_up_gate,
        bool parallel_stage) {
    const char *path = std::getenv("GGML_MOE_UP_GATE_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,mode,up_tensor,gate_tensor,up_type,gate_type,n_active,"
                "up_cache_hits,up_cache_misses,gate_cache_hits,gate_cache_misses,"
                "up_stage_jobs,gate_stage_jobs,stage_ms,quant_ms,up_ms,gate_ms,"
                "up_wait_ms,gate_wait_ms,up_compute_ms,gate_compute_ms,fuse_ms,"
                "kernel_ms,d2h_ms,scatter_ms,wall_ms,use_handoff,parallel_up_gate,parallel_stage\n");
        header_written = true;
    }

    static uint64_t seq = 0;
    std::fprintf(f,
            "%lu,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n",
            (unsigned long)++seq,
            prompt_mode ? "prompt" : "decode",
            up_tensor ? up_tensor : "",
            gate_tensor ? gate_tensor : "",
            (int)up_type,
            (int)gate_type,
            n_active,
            up_cache_hits,
            up_cache_misses,
            gate_cache_hits,
            gate_cache_misses,
            up_stage_jobs,
            gate_stage_jobs,
            (double)stage_ms,
            (double)quant_ms,
            (double)up_ms,
            (double)gate_ms,
            (double)up_wait_ms,
            (double)gate_wait_ms,
            (double)up_compute_ms,
            (double)gate_compute_ms,
            (double)fuse_ms,
            (double)kernel_ms,
            (double)d2h_ms,
            scatter_ms,
            wall_ms,
            use_handoff ? 1 : 0,
            parallel_up_gate ? 1 : 0,
            parallel_stage ? 1 : 0);
    std::fclose(f);
}

static void current_down_overlap_atomic_max(std::atomic<uint64_t> &target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
            !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static void current_down_overlap_record_batch(size_t jobs) {
    if (jobs == 0) return;
    ++g_current_down_overlap.submitted_batches;
    g_current_down_overlap.planned_jobs.fetch_add(jobs);
    current_down_overlap_atomic_max(g_current_down_overlap.max_jobs, jobs);
    switch (expert_pack_iouring_batch_bucket(jobs)) {
        case 0: ++g_current_down_overlap.batch_hist_1; break;
        case 1: ++g_current_down_overlap.batch_hist_2_4; break;
        case 2: ++g_current_down_overlap.batch_hist_5_8; break;
        case 3: ++g_current_down_overlap.batch_hist_9_16; break;
        case 4: ++g_current_down_overlap.batch_hist_17_32; break;
        default: ++g_current_down_overlap.batch_hist_gt32; break;
    }
}

static bool current_down_overlap_enabled() {
    const char *env = std::getenv("GGML_MOE_CURRENT_DOWN_OVERLAP");
    return env && env[0] && env[0] != '0';
}

static bool current_down_overlap_tensor_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_CURRENT_DOWN_OVERLAP_PROFILE_OUT");
    return env && env[0];
}

static void current_down_overlap_tensor_profile_record(
        const char *tensor,
        uint64_t calls,
        uint64_t planned_jobs,
        uint64_t cache_hits,
        uint64_t missing_tensor,
        uint64_t missing_pack,
        uint64_t completed_jobs) {
    if (!current_down_overlap_tensor_profile_enabled()) return;
    const char *key = (tensor && tensor[0]) ? tensor : "<unknown>";
    std::lock_guard<std::mutex> lk(g_current_down_overlap_tensor_profile_mu);
    current_down_overlap_tensor_profile &p = g_current_down_overlap_tensor_profile[key];
    p.calls += calls;
    p.planned_jobs += planned_jobs;
    p.cache_hits += cache_hits;
    p.missing_tensor += missing_tensor;
    p.missing_pack += missing_pack;
    p.completed_jobs += completed_jobs;
}

static void current_down_overlap_tensor_profile_write() {
    const char *path = std::getenv("GGML_MOE_CURRENT_DOWN_OVERLAP_PROFILE_OUT");
    if (!path || !path[0]) return;

    std::lock_guard<std::mutex> lk(g_current_down_overlap_tensor_profile_mu);
    if (g_current_down_overlap_tensor_profile.empty()) return;

    FILE *f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f, "tensor,calls,planned_jobs,cache_hits,missing_tensor,missing_pack,completed_jobs\n");
    for (const auto &it : g_current_down_overlap_tensor_profile) {
        const current_down_overlap_tensor_profile &p = it.second;
        std::fprintf(f, "%s,%lu,%lu,%lu,%lu,%lu,%lu\n",
                it.first.c_str(),
                (unsigned long)p.calls,
                (unsigned long)p.planned_jobs,
                (unsigned long)p.cache_hits,
                (unsigned long)p.missing_tensor,
                (unsigned long)p.missing_pack,
                (unsigned long)p.completed_jobs);
    }
    std::fclose(f);
}

static void current_down_overlap_report_atexit() {
    const uint64_t calls = g_current_down_overlap.calls.load();
    const uint64_t submitted = g_current_down_overlap.submitted_batches.load();
    if (calls == 0 && submitted == 0) {
        current_down_overlap_tensor_profile_write();
        return;
    }
    std::fprintf(stderr,
        "[moe_stream_batch] current down overlap: calls=%lu planned_jobs=%lu completed_jobs=%lu "
        "cache_hits=%lu missing_tensor=%lu missing_pack=%lu submitted_batches=%lu failed_batches=%lu "
        "mark_failed=%lu max_jobs=%lu worker_us=%lu batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
        calls,
        g_current_down_overlap.planned_jobs.load(),
        g_current_down_overlap.completed_jobs.load(),
        g_current_down_overlap.cache_hits.load(),
        g_current_down_overlap.missing_tensor.load(),
        g_current_down_overlap.missing_pack.load(),
        submitted,
        g_current_down_overlap.failed_batches.load(),
        g_current_down_overlap.mark_failed.load(),
        g_current_down_overlap.max_jobs.load(),
        g_current_down_overlap.worker_us.load(),
        g_current_down_overlap.batch_hist_1.load(),
        g_current_down_overlap.batch_hist_2_4.load(),
        g_current_down_overlap.batch_hist_5_8.load(),
        g_current_down_overlap.batch_hist_9_16.load(),
        g_current_down_overlap.batch_hist_17_32.load(),
        g_current_down_overlap.batch_hist_gt32.load());
    current_down_overlap_tensor_profile_write();
}

static void expert_pack_report_atexit() {
    if (!g_expert_pack.enabled) return;
    std::fprintf(stderr,
                 "[moe_stream_batch] expert pack: hits=%lu misses=%lu read_failures=%lu direct_reads=%lu direct_fallbacks=%lu "
                 "iouring_reads=%lu iouring_bytes=%lu iouring_fallbacks=%lu iouring_submit_us=%lu iouring_wait_us=%lu iouring_h2d_enqueues=%lu entries=%zu\n",
                 g_expert_pack.hits.load(), g_expert_pack.misses.load(),
                 g_expert_pack.read_failures.load(), g_expert_pack.direct_reads.load(),
                 g_expert_pack.direct_fallbacks.load(),
                 g_expert_pack.iouring_reads.load(), g_expert_pack.iouring_bytes.load(),
                 g_expert_pack.iouring_fallbacks.load(), g_expert_pack.iouring_submit_us.load(),
                 g_expert_pack.iouring_wait_us.load(), g_expert_pack.iouring_h2d_enqueues.load(),
                 g_expert_pack.entries.size());
    const uint64_t inflight_samples = g_expert_pack.iouring_inflight_samples.load();
    const double inflight_avg = inflight_samples > 0 ?
        (double)g_expert_pack.iouring_inflight_sum.load() / (double)inflight_samples : 0.0;
    if (g_expert_pack.iouring_batches.load() > 0 ||
            g_expert_pack.iouring_submit_calls.load() > 0 ||
            g_expert_pack.iouring_wait_calls.load() > 0) {
        std::fprintf(stderr,
                     "[moe_stream_batch] expert pack iouring detail: batches=%lu submit_calls=%lu wait_calls=%lu cqes=%lu "
                     "inflight_avg=%.2f inflight_max=%lu batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
                     g_expert_pack.iouring_batches.load(), g_expert_pack.iouring_submit_calls.load(),
                     g_expert_pack.iouring_wait_calls.load(), g_expert_pack.iouring_cqes.load(),
                     inflight_avg, g_expert_pack.iouring_inflight_max.load(),
                     g_expert_pack.iouring_batch_hist_1.load(), g_expert_pack.iouring_batch_hist_2_4.load(),
                     g_expert_pack.iouring_batch_hist_5_8.load(), g_expert_pack.iouring_batch_hist_9_16.load(),
                     g_expert_pack.iouring_batch_hist_17_32.load(), g_expert_pack.iouring_batch_hist_gt32.load());
    }
    if (g_expert_pack.ram_tier_base) {
        std::fprintf(stderr,
                     "[moe_stream_batch] RAM tier: hits=%lu total=%lu hit_rate=%.1f%% resident=%.2f MiB\n",
                     g_expert_pack.ram_tier_hits.load(), g_expert_pack.ram_tier_total.load(),
                     g_expert_pack.ram_tier_total.load() > 0 ?
                         100.0 * g_expert_pack.ram_tier_hits.load() / g_expert_pack.ram_tier_total.load() : 0.0,
                     g_expert_pack.ram_tier_bytes / (1024.0 * 1024.0));
    }
}

extern "C" void ggml_cuda_moe_stream_batch_report_counters(void) {
    batch_cache_report_atexit();
    expert_pack_report_atexit();
    std::fflush(stderr);
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

static expert_pack_source * expert_pack_source_for_entry(const expert_pack_entry *entry) {
    if (!entry || entry->source_idx < 0 ||
            (size_t)entry->source_idx >= g_expert_pack.sources.size()) {
        return nullptr;
    }
    return &g_expert_pack.sources[(size_t)entry->source_idx];
}

static bool expert_pack_load_source(const char *path, int32_t source_idx, std::vector<expert_pack_entry> &entries) {
    if (!path || !path[0]) return false;

    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: open failed: %s\n", path);
        return false;
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
        return false;
    }

    expert_pack_source source;
    source.file = file;
    std::snprintf(source.path, sizeof(source.path), "%s", path);
#if !defined(_WIN32)
    source.fd_direct = -1;
    if (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) {
#if defined(O_DIRECT)
        source.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
#endif
        if (source.fd_direct < 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: direct open failed; using buffered reads: %s\n", path);
            g_expert_pack.io_backend = 0;
        } else if (g_expert_pack.io_backend == 2) {
#if defined(GGML_MOE_HAS_LIBURING)
            std::fprintf(stderr, "[moe_stream_batch] expert pack: io_uring direct reads enabled: %s io_bytes=%zu depth=%zu\n",
                         path, expert_pack_io_bytes(), expert_pack_io_depth());
#else
            const bool require_direct = expert_pack_env_bool("GGML_MOE_IO_REQUIRE_DIRECT", false);
            std::fprintf(stderr, "[moe_stream_batch] expert pack: io_uring requested but liburing headers are unavailable; %s\n",
                         require_direct ? "using direct fallback anyway" : "using direct fallback");
            g_expert_pack.io_backend = 1;
#endif
        } else if (g_expert_pack.io_backend == 1) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: direct reads enabled: %s\n", path);
        }
    }
#else
    if (g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: direct/io_uring reads are not supported on this platform; using buffered reads\n");
        g_expert_pack.io_backend = 0;
    }
#endif

    if ((size_t)source_idx != g_expert_pack.sources.size()) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: internal source index mismatch for %s\n", path);
        std::fclose(file);
        return false;
    }
    g_expert_pack.sources.push_back(source);

    const size_t before = entries.size();
    entries.resize(before + (size_t)n_entries);
    for (uint64_t i = 0; i < n_entries; ++i) {
        expert_pack_entry &e = entries[before + (size_t)i];
        uint32_t reserved = 0;
        if (!expert_pack_read_exact(file, e.tensor, sizeof(e.tensor)) ||
                !expert_pack_read_exact(file, &e.expert_idx, sizeof(e.expert_idx)) ||
                !expert_pack_read_exact(file, &reserved, sizeof(reserved)) ||
                !expert_pack_read_exact(file, &e.offset, sizeof(e.offset)) ||
                !expert_pack_read_exact(file, &e.nbytes, sizeof(e.nbytes))) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: short index: %s\n", path);
            return false;
        }
        e.tensor[sizeof(e.tensor) - 1] = '\0';
        e.source_idx = source_idx;
    }

    std::fprintf(stderr, "[moe_stream_batch] expert pack: loaded %lu entries from %s\n",
                 (unsigned long)n_entries, path);
    return true;
}

static void expert_pack_init_once() {
    std::lock_guard<std::mutex> lk(g_expert_pack.mu);
    if (g_expert_pack.inited) return;

    const char *io_backend_env = std::getenv("GGML_MOE_IO_BACKEND");
    if (io_backend_env && io_backend_env[0]) {
        if (std::strcmp(io_backend_env, "direct") == 0) {
            g_expert_pack.io_backend = 1;
        } else if (std::strcmp(io_backend_env, "iouring") == 0 ||
                   std::strcmp(io_backend_env, "io_uring") == 0) {
            g_expert_pack.io_backend = 2;
        } else if (std::strcmp(io_backend_env, "mmap") == 0) {
            g_expert_pack.io_backend = 0;
        } else if (!g_expert_pack.reported_io_backend) {
            std::fprintf(stderr,
                "[moe_stream_batch] GGML_MOE_IO_BACKEND=%s requested; runtime supports mmap/buffered, direct, and iouring expert-pack reads\n",
                io_backend_env);
            g_expert_pack.reported_io_backend = true;
        }
    }

    const char *path = std::getenv("GGML_MOE_EXPERT_PACK");
    if (!path || !path[0]) {
        g_expert_pack.inited = true;
        return;
    }

    std::vector<expert_pack_entry> entries;
    if (!expert_pack_load_source(path, 0, entries)) {
        g_expert_pack.inited = true;
        return;
    }
    const char *overlay_path = std::getenv("GGML_MOE_EXPERT_PACK_OVERLAY");
    if (overlay_path && overlay_path[0]) {
        if (!expert_pack_load_source(overlay_path, (int32_t)g_expert_pack.sources.size(), entries)) {
            g_expert_pack.inited = true;
            return;
        }
    }

    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_entry &a, const expert_pack_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            return a.nbytes < b.nbytes;
        });
    for (size_t i = 1; i < entries.size(); ++i) {
        if (expert_pack_entry_cmp(entries[i - 1], entries[i].tensor, entries[i].expert_idx, entries[i].nbytes) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: duplicate key across packs: %s expert=%d bytes=%lu\n",
                         entries[i].tensor, entries[i].expert_idx, (unsigned long)entries[i].nbytes);
            g_expert_pack.inited = true;
            return;
        }
    }

    g_expert_pack.file = g_expert_pack.sources.empty() ? nullptr : g_expert_pack.sources[0].file;
#if !defined(_WIN32)
    g_expert_pack.fd_direct = g_expert_pack.sources.empty() ? -1 : g_expert_pack.sources[0].fd_direct;
#endif
    g_expert_pack.entries = std::move(entries);
    g_expert_pack.enabled = true;
    g_expert_pack.inited = true;
    std::atexit(expert_pack_report_atexit);
    std::fprintf(stderr, "[moe_stream_batch] expert pack: total entries=%zu sources=%zu\n",
                 g_expert_pack.entries.size(), g_expert_pack.sources.size());
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


static bool expert_pack_mmap_ensure() {
    expert_pack_init_once();
    if (!g_expert_pack.enabled || g_expert_pack.mmap_base) {
        return g_expert_pack.mmap_base != nullptr;
    }
    if (g_expert_pack.mmap_attempted) {
        return false;
    }
    g_expert_pack.mmap_attempted = true;
#if !defined(_WIN32)
    const char *env = std::getenv("GGML_MOE_CPU_FALLBACK_PACK_MMAP");
    if (!env || !env[0] || env[0] == '0') {
        return false;
    }
    if (!g_expert_pack.file) {
        return false;
    }
    const int fd = fileno(g_expert_pack.file);
    if (fd < 0) {
        return false;
    }
    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        return false;
    }
    void *base = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack mmap: mmap failed\n");
        return false;
    }
    g_expert_pack.mmap_base = base;
    g_expert_pack.mmap_size = (size_t) st.st_size;
    g_expert_pack.mmap_enabled = true;
    std::fprintf(stderr, "[moe_stream_batch] expert pack mmap: enabled size=%.2f MiB\n",
                 g_expert_pack.mmap_size / (1024.0 * 1024.0));
    return true;
#else
    return false;
#endif
}

extern "C" const void * ggml_cuda_moe_expert_pack_mmap_ptr(const char *tensor_name, int expert_idx, size_t nbytes) {
    if (!tensor_name || !tensor_name[0] || nbytes == 0) {
        return nullptr;
    }
    if (!expert_pack_mmap_ensure()) {
        return nullptr;
    }
    const expert_pack_entry *entry = expert_pack_lookup(tensor_name, expert_idx, nbytes);
    if (!entry) {
        ++g_expert_pack.mmap_misses;
        return nullptr;
    }
    if (entry->offset > g_expert_pack.mmap_size || entry->nbytes != nbytes || entry->offset + entry->nbytes > g_expert_pack.mmap_size) {
        ++g_expert_pack.mmap_misses;
        return nullptr;
    }
    ++g_expert_pack.mmap_hits;
    g_expert_pack.mmap_bytes.fetch_add(nbytes);
    return (const char *) g_expert_pack.mmap_base + entry->offset;
}

static bool trace_entry_matches(const batch_route_trace_entry &e, const char *tensor_name, int expert_idx, size_t expert_bytes);

static uint64_t host_prefetch_key(const char *tensor_name, int expert_idx, size_t nbytes) {
    uint64_t h = 1469598103934665603ULL;
    if (tensor_name) {
        for (const unsigned char *p = (const unsigned char *)tensor_name; *p; ++p) {
            h ^= (uint64_t)(*p);
            h *= 1099511628211ULL;
        }
    }
    h ^= (uint64_t)(uint32_t)expert_idx;
    h *= 1099511628211ULL;
    h ^= (uint64_t)nbytes;
    h *= 1099511628211ULL;
    return h;
}

static void host_prefetch_report_atexit() {
    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    if (!g_host_prefetch.enabled) return;
    std::fprintf(stderr,
        "[moe_stream_batch] host prefetch: calls=%lu matched=%lu resync=%lu submitted=%lu hits=%lu misses=%lu "
        "evicted=%lu read_failures=%lu alloc_failures=%lu duplicate_skips=%lu profile_pinned_skips=%lu reserved_skips=%lu no_slot=%lu "
        "planned_enqueued=%lu planned_dequeued=%lu planned_duplicate_skips=%lu scan_passes=%lu "
        "cursor=%zu produce_cursor=%zu/%zu skip=%zu used=%.2f MiB slots=%zu\n",
        g_host_prefetch.calls,
        g_host_prefetch.matched,
        g_host_prefetch.resync,
        g_host_prefetch.submitted,
        g_host_prefetch.hits,
        g_host_prefetch.misses,
        g_host_prefetch.evicted,
        g_host_prefetch.read_failures,
        g_host_prefetch.alloc_failures,
        g_host_prefetch.duplicate_skips,
        g_host_prefetch.profile_pinned_skips,
        g_host_prefetch.reserved_skips,
        g_host_prefetch.no_slot,
        g_host_prefetch.planned_enqueued,
        g_host_prefetch.planned_dequeued,
        g_host_prefetch.planned_duplicate_skips,
        g_host_prefetch.scan_passes,
        g_host_prefetch.cursor,
        g_host_prefetch.produce_cursor,
        g_host_prefetch.trace.size(),
        g_host_prefetch.skip_events,
        g_host_prefetch.used_bytes / (1024.0 * 1024.0),
        g_host_prefetch.slots.size());
}

static int host_prefetch_env_int(const char *name, int fallback, int lo, int hi) {
    const char *env = std::getenv(name);
    long value = (env && env[0]) ? std::atol(env) : fallback;
    if (value < lo) value = lo;
    if (value > hi) value = hi;
    return (int)value;
}

static size_t host_prefetch_env_mib(const char *name, size_t fallback_mib, size_t lo_mib, size_t hi_mib) {
    const char *env = std::getenv(name);
    size_t value = (env && env[0]) ? (size_t)std::strtoull(env, nullptr, 10) : fallback_mib;
    if (value < lo_mib) value = lo_mib;
    if (value > hi_mib) value = hi_mib;
    return value * 1024ULL * 1024ULL;
}

static bool host_prefetch_slot_alloc(host_prefetch_slot &slot, size_t nbytes) {
    if (slot.capacity >= nbytes && slot.host) return true;
    if (slot.host) {
        cudaFreeHost(slot.host);
        slot.host = nullptr;
        slot.capacity = 0;
    }
    const size_t alloc_sz = (size_t)align_up_u64((uint64_t)nbytes, (uint64_t)expert_pack_direct_alignment());
    if (cudaHostAlloc(&slot.host, alloc_sz, cudaHostAllocDefault) != cudaSuccess) {
        slot.host = nullptr;
        slot.capacity = 0;
        return false;
    }
    if (!slot.done && cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming) != cudaSuccess) {
        cudaFreeHost(slot.host);
        slot.host = nullptr;
        slot.capacity = 0;
        return false;
    }
    slot.capacity = alloc_sz;
    return true;
}

static int host_prefetch_find_free_slot_locked(size_t nbytes) {
    int best = -1;
    for (size_t i = 0; i < g_host_prefetch.slots.size(); ++i) {
        const host_prefetch_slot &slot = g_host_prefetch.slots[i];
        if (slot.in_use) continue;
        if (slot.reserved) continue;
        if (!slot.ready) {
            best = (int)i;
            break;
        }
        if (best < 0) best = (int)i;
    }
    if (best >= 0 && g_host_prefetch.slots[(size_t)best].ready) {
        host_prefetch_slot &slot = g_host_prefetch.slots[(size_t)best];
        g_host_prefetch.ready.erase(host_prefetch_key(slot.tensor, slot.expert_idx, slot.nbytes));
        slot.ready = false;
        slot.reserved = false;
        slot.entry = nullptr;
        ++g_host_prefetch.evicted;
    }
    const size_t alloc_sz = (size_t)align_up_u64((uint64_t)nbytes, (uint64_t)expert_pack_direct_alignment());
    const size_t cur_capacity = best >= 0 ? g_host_prefetch.slots[(size_t)best].capacity : 0;
    const size_t extra = alloc_sz > cur_capacity ? alloc_sz - cur_capacity : 0;
    if (best >= 0 && g_host_prefetch.used_bytes + extra > g_host_prefetch.max_bytes) {
        return -1;
    }
    return best;
}

static void host_prefetch_worker() {
    while (true) {
        batch_route_trace_entry e;
        size_t slot_idx = SIZE_MAX;
        const expert_pack_entry *entry = nullptr;
        {
            std::unique_lock<std::mutex> lk(g_host_prefetch.mu);
            g_host_prefetch.cv.wait(lk, [] {
                return g_host_prefetch.stop ||
                    (g_host_prefetch.enabled &&
                     (!g_host_prefetch.planned.empty() ||
                      g_host_prefetch.cursor < g_host_prefetch.trace.size()));
            });
            if (g_host_prefetch.stop) break;

            ++g_host_prefetch.scan_passes;
            if (!g_host_prefetch.planned.empty()) {
                e = g_host_prefetch.planned.front();
                g_host_prefetch.planned.pop_front();
                const uint64_t key = host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes);
                g_host_prefetch.planned_queued.erase(key);
                ++g_host_prefetch.planned_dequeued;
            } else {
                const size_t start = std::max(g_host_prefetch.cursor, g_host_prefetch.skip_events);
                const size_t end = std::min(g_host_prefetch.trace.size(), start + g_host_prefetch.lead_events);
                size_t selected = SIZE_MAX;
                for (size_t i = start; i < end; ++i) {
                    const batch_route_trace_entry &candidate = g_host_prefetch.trace[i];
                    const uint64_t key = host_prefetch_key(candidate.tensor, candidate.expert_idx, candidate.expert_bytes);
                    if (profile_pinned_key_contains(batch_key_hash(candidate.tensor, candidate.expert_idx))) {
                        ++g_host_prefetch.profile_pinned_skips;
                        continue;
                    }
                    auto it = g_host_prefetch.ready.find(key);
                    if (it != g_host_prefetch.ready.end()) {
                        const host_prefetch_slot &slot = g_host_prefetch.slots[it->second];
                        if (slot.ready) {
                            ++g_host_prefetch.duplicate_skips;
                        } else {
                            ++g_host_prefetch.reserved_skips;
                        }
                        continue;
                    }
                    selected = i;
                    break;
                }
                if (selected == SIZE_MAX) {
                    g_host_prefetch.cv.wait_for(lk, std::chrono::milliseconds(1));
                    continue;
                }

                e = g_host_prefetch.trace[selected];
                g_host_prefetch.produce_cursor = selected + 1;
            }
            const uint64_t key = host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes);
            entry = expert_pack_lookup(e.tensor, e.expert_idx, e.expert_bytes);
            if (!entry) {
                ++g_host_prefetch.read_failures;
                continue;
            }
            const int free_slot = host_prefetch_find_free_slot_locked(e.expert_bytes);
            if (free_slot < 0) {
                ++g_host_prefetch.no_slot;
                continue;
            }
            slot_idx = (size_t)free_slot;
            host_prefetch_slot &slot = g_host_prefetch.slots[slot_idx];
            slot.in_use = true;
            slot.reserved = true;
            slot.ready = false;
            slot.nbytes = e.expert_bytes;
            slot.expert_idx = e.expert_idx;
            std::snprintf(slot.tensor, sizeof(slot.tensor), "%s", e.tensor);
            g_host_prefetch.ready[key] = slot_idx;
        }

        bool ok = false;
        host_prefetch_slot *slot_ptr = &g_host_prefetch.slots[slot_idx];
        const size_t old_capacity = slot_ptr->capacity;
        if (host_prefetch_slot_alloc(*slot_ptr, e.expert_bytes)) {
            if (slot_ptr->done) {
                cudaEventSynchronize(slot_ptr->done);
            }
            const expert_pack_source *source = expert_pack_source_for_entry(entry);
            if (source && source->fd_direct >= 0 &&
                    ((entry->offset % expert_pack_direct_alignment()) == 0) &&
                    (((uintptr_t)slot_ptr->host % expert_pack_direct_alignment()) == 0)) {
                const size_t read_sz = (size_t)align_up_u64((uint64_t)e.expert_bytes, (uint64_t)expert_pack_direct_alignment());
                ok = ::pread(source->fd_direct, slot_ptr->host, read_sz, (off_t)entry->offset) == (ssize_t)read_sz;
            } else if (source && source->file) {
                std::lock_guard<std::mutex> lk(g_expert_pack.mu);
                ok = ::fseeko(source->file, (off_t)entry->offset, SEEK_SET) == 0 &&
                    expert_pack_read_exact(source->file, slot_ptr->host, e.expert_bytes);
            }
        } else {
            std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
            ++g_host_prefetch.alloc_failures;
        }

        {
            std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
            host_prefetch_slot &slot = g_host_prefetch.slots[slot_idx];
            if (slot.capacity > old_capacity) {
                g_host_prefetch.used_bytes += slot.capacity - old_capacity;
            } else if (slot.capacity < old_capacity) {
                const size_t delta = old_capacity - slot.capacity;
                g_host_prefetch.used_bytes = delta < g_host_prefetch.used_bytes ? g_host_prefetch.used_bytes - delta : 0;
            }
            if (ok) {
                slot.entry = entry;
                slot.nbytes = e.expert_bytes;
                slot.expert_idx = e.expert_idx;
                std::snprintf(slot.tensor, sizeof(slot.tensor), "%s", e.tensor);
                slot.reserved = false;
                slot.ready = true;
                g_host_prefetch.ready[host_prefetch_key(slot.tensor, slot.expert_idx, slot.nbytes)] = slot_idx;
                ++g_host_prefetch.submitted;
            } else {
                slot.ready = false;
                slot.reserved = false;
                slot.entry = nullptr;
                g_host_prefetch.ready.erase(host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes));
                ++g_host_prefetch.read_failures;
            }
            slot.in_use = false;
        }
    }
}

static void host_prefetch_shutdown() {
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        g_host_prefetch.stop = true;
        g_host_prefetch.cv.notify_all();
    }
    if (g_host_prefetch.worker.joinable()) {
        g_host_prefetch.worker.join();
    }
    for (host_prefetch_slot &slot : g_host_prefetch.slots) {
        if (slot.host) {
            if (slot.done) {
                cudaEventSynchronize(slot.done);
            }
            cudaFreeHost(slot.host);
            slot.host = nullptr;
        }
        if (slot.done) {
            cudaEventDestroy(slot.done);
            slot.done = nullptr;
        }
    }
}

static void host_prefetch_init_once() {
    if (g_host_prefetch.inited) return;
    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    if (g_host_prefetch.inited) return;
    const char *path = std::getenv("GGML_MOE_HOST_PREFETCH");
    const char *planned_env = std::getenv("GGML_MOE_PLANNED_HOST_PREFETCH");
    g_host_prefetch.planned_enabled = planned_env && planned_env[0] && planned_env[0] != '0';
    if ((!path || !path[0]) && !g_host_prefetch.planned_enabled) {
        g_host_prefetch.inited = true;
        return;
    }
    if (path && path[0] && !load_route_trace_csv(path, g_host_prefetch.trace, "host prefetch")) {
        g_host_prefetch.inited = true;
        return;
    }
    const int slots = host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_SLOTS", 64, 1, 512);
    g_host_prefetch.lead_events = (size_t)host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_LEAD_EVENTS", 2048, 1, 32768);
    g_host_prefetch.skip_events = (size_t)host_prefetch_env_int("GGML_MOE_HOST_PREFETCH_SKIP_EVENTS", 0, 0, 100000000);
    if (g_host_prefetch.skip_events > g_host_prefetch.trace.size()) {
        g_host_prefetch.skip_events = g_host_prefetch.trace.size();
    }
    g_host_prefetch.produce_cursor = g_host_prefetch.skip_events;
    g_host_prefetch.max_bytes = host_prefetch_env_mib("GGML_MOE_HOST_PREFETCH_MAX_MIB", 512, 16, 8192);
    g_host_prefetch.slots.resize((size_t)slots);
    g_host_prefetch.enabled = true;
    g_host_prefetch.inited = true;
    std::atexit(host_prefetch_report_atexit);
    std::atexit(host_prefetch_shutdown);
    g_host_prefetch.worker = std::thread(host_prefetch_worker);
    g_host_prefetch.cv.notify_all();
    std::fprintf(stderr,
        "[moe_stream_batch] host prefetch: loaded %zu events from %s lead_events=%zu skip_events=%zu slots=%d max=%.2f MiB planned=%d\n",
        g_host_prefetch.trace.size(), (path && path[0]) ? path : "(none)",
        g_host_prefetch.lead_events, g_host_prefetch.skip_events, slots,
        g_host_prefetch.max_bytes / (1024.0 * 1024.0),
        g_host_prefetch.planned_enabled ? 1 : 0);
}

static void host_prefetch_on_route(const char *tensor_name, int expert_idx, size_t expert_bytes) {
    host_prefetch_init_once();
    if (!g_host_prefetch.enabled || !tensor_name || !tensor_name[0]) return;
    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    ++g_host_prefetch.calls;
    if (g_host_prefetch.cursor < g_host_prefetch.trace.size() &&
            trace_entry_matches(g_host_prefetch.trace[g_host_prefetch.cursor], tensor_name, expert_idx, expert_bytes)) {
        ++g_host_prefetch.cursor;
        ++g_host_prefetch.matched;
    } else {
        const size_t scan_end = std::min(g_host_prefetch.trace.size(), g_host_prefetch.cursor + (size_t)128);
        for (size_t i = g_host_prefetch.cursor; i < scan_end; ++i) {
            if (trace_entry_matches(g_host_prefetch.trace[i], tensor_name, expert_idx, expert_bytes)) {
                g_host_prefetch.cursor = i + 1;
                ++g_host_prefetch.matched;
                ++g_host_prefetch.resync;
                break;
            }
        }
    }
    if (g_host_prefetch.produce_cursor < g_host_prefetch.cursor) {
        g_host_prefetch.produce_cursor = g_host_prefetch.cursor;
    }
    g_host_prefetch.cv.notify_one();
}

static void host_prefetch_submit_planned(const char *tensor_name, int expert_idx, size_t expert_bytes) {
    host_prefetch_init_once();
    if (!g_host_prefetch.enabled || !g_host_prefetch.planned_enabled ||
            !tensor_name || !tensor_name[0] || expert_idx < 0 || expert_bytes == 0) {
        return;
    }
    batch_route_trace_entry e;
    e.seq = 0;
    e.expert_idx = expert_idx;
    e.expert_bytes = expert_bytes;
    std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor_name);

    std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
    const uint64_t key = host_prefetch_key(e.tensor, e.expert_idx, e.expert_bytes);
    auto ready_it = g_host_prefetch.ready.find(key);
    if (ready_it != g_host_prefetch.ready.end()) {
        ++g_host_prefetch.planned_duplicate_skips;
        return;
    }
    if (g_host_prefetch.planned_queued.find(key) != g_host_prefetch.planned_queued.end()) {
        ++g_host_prefetch.planned_duplicate_skips;
        return;
    }
    g_host_prefetch.planned_queued[key] = g_host_prefetch.planned.size();
    g_host_prefetch.planned.push_back(e);
    ++g_host_prefetch.planned_enqueued;
    g_host_prefetch.cv.notify_one();
}

static bool host_prefetch_copy_h2d(
        const expert_pack_entry *pack_entry,
        const char *tensor_name,
        int expert_idx,
        void *dst,
        size_t sz,
        cudaStream_t st) {
    host_prefetch_init_once();
    if (!g_host_prefetch.enabled || !pack_entry || !tensor_name || !tensor_name[0]) return false;
    void *host = nullptr;
    size_t slot_idx = SIZE_MAX;
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        const uint64_t key = host_prefetch_key(tensor_name, expert_idx, sz);
        auto it = g_host_prefetch.ready.find(key);
        if (it == g_host_prefetch.ready.end()) {
            ++g_host_prefetch.misses;
            return false;
        }
        host_prefetch_slot &slot = g_host_prefetch.slots[it->second];
        if (!slot.ready || slot.nbytes != sz || !slot.host) {
            if (slot.reserved) {
                ++g_host_prefetch.reserved_skips;
            } else {
                g_host_prefetch.ready.erase(it);
            }
            ++g_host_prefetch.misses;
            return false;
        }
        slot_idx = it->second;
        host = slot.host;
        slot.in_use = true;
        slot.ready = false;
        slot.reserved = false;
        slot.entry = nullptr;
        g_host_prefetch.ready.erase(it);
        ++g_host_prefetch.hits;
    }
    const bool ok = host && cudaMemcpyAsync(dst, host, sz, cudaMemcpyHostToDevice, st) == cudaSuccess;
    {
        std::lock_guard<std::mutex> lk(g_host_prefetch.mu);
        if (slot_idx < g_host_prefetch.slots.size()) {
            if (ok && g_host_prefetch.slots[slot_idx].done) {
                cudaEventRecord(g_host_prefetch.slots[slot_idx].done, st);
            }
            g_host_prefetch.slots[slot_idx].in_use = false;
        }
    }
    return ok;
}

static std::once_flag g_ram_tier_once;

static void ram_tier_init() {
    if (!g_expert_pack.enabled) return;
    const char *ram_tier_env = std::getenv("GGML_MOE_RAM_TIER_MIB");
    if (!ram_tier_env || !ram_tier_env[0] || ram_tier_env[0] == '0') return;
    const size_t budget = (size_t)std::atol(ram_tier_env) * 1024ULL * 1024ULL;
    if (budget == 0) return;

    std::vector<profile_entry> ram_profile;
    const char *ram_profile_path = std::getenv("GGML_MOE_RAM_TIER_PROFILE");
    if (ram_profile_path && ram_profile_path[0]) {
        if (!load_profile_file(ram_profile_path, ram_profile, "RAM tier profile")) {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: no RAM tier profile loaded, skipping\n");
            return;
        }
    } else {
        load_profile_once();
        ram_profile = g_profile;
    }
    if (ram_profile.empty()) {
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
    const char *pin_mib_env = std::getenv("GGML_MOE_RAM_TIER_PIN_MIB");
    const bool pin_budget_set = pin_mib_env && pin_mib_env[0];
    const size_t pin_budget = pin_budget_set ?
        (size_t)std::atol(pin_mib_env) * 1024ULL * 1024ULL : budget;

    size_t loaded = 0, n_loaded = 0, n_skipped = 0, pin_prefix_bytes = 0, pin_prefix_entries = 0;
    for (const profile_entry &pe : ram_profile) {
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
            const expert_pack_source *source = expert_pack_source_for_entry(&ent);
            if (!source || !source->file) continue;
            if (::fseeko(source->file, (off_t)ent.offset, SEEK_SET) != 0) continue;
            if (!expert_pack_read_exact(source->file, dst_ptr, (size_t)ent.nbytes)) continue;
        }
        g_expert_pack.ram_tier_index[lo2] = {loaded, (size_t)ent.nbytes};
        loaded += (size_t)ent.nbytes;
        ++n_loaded;
        if (pin_prefix_bytes + (size_t)ent.nbytes <= pin_budget) {
            pin_prefix_bytes += (size_t)ent.nbytes;
            ++pin_prefix_entries;
        }
    }
    std::fprintf(stderr, "[moe_stream_batch] RAM tier: loaded %zu entries (skipped %zu), %.2f MiB into anonymous mmap\n",
                 n_loaded, skip, loaded / (1024.0 * 1024.0));
    if (loaded > 0 && expert_pack_env_bool("GGML_MOE_RAM_TIER_PIN", true)) {
        const size_t pin_bytes = std::min(loaded, pin_prefix_bytes);
        if (pin_bytes == 0) {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister prefix budget is 0, using unpinned path (%.2f MiB)\n",
                         loaded / (1024.0 * 1024.0));
            return;
        }
        cudaError_t err = cudaHostRegister(region, pin_bytes, cudaHostRegisterDefault);
        if (err == cudaSuccess) {
            g_expert_pack.ram_tier_pinned_bytes = pin_bytes;
            if (pin_budget_set) {
                std::fprintf(stderr,
                             "[moe_stream_batch] RAM tier: cudaHostRegister succeeded (%.2f MiB pinned prefix, %zu entries, %.2f MiB resident)\n",
                             pin_bytes / (1024.0 * 1024.0), pin_prefix_entries, loaded / (1024.0 * 1024.0));
            } else {
                std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister succeeded (%.2f MiB pinned)\n",
                             pin_bytes / (1024.0 * 1024.0));
            }
        } else {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister failed (%s), using unpinned path\n",
                         cudaGetErrorString(err));
        }
    } else if (loaded > 0) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister disabled, using unpinned path (%.2f MiB)\n",
                     loaded / (1024.0 * 1024.0));
    }
}

static bool expert_pack_read_entry_iouring(const expert_pack_entry *entry, void *dst, size_t sz) {
#if defined(GGML_MOE_HAS_LIBURING) && !defined(_WIN32)
    expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (!entry || !source || source->fd_direct < 0 || entry->nbytes != sz) return false;
    const uint64_t alignment = expert_pack_direct_alignment();
    const size_t read_sz = (size_t)align_up_u64((uint64_t)sz, alignment);
    if ((entry->offset % alignment) != 0 ||
            ((uintptr_t)dst % alignment) != 0 ||
            read_sz < sz) {
        ++g_expert_pack.iouring_fallbacks;
        return false;
    }

    io_uring ring_io;
    const unsigned int flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ? IORING_SETUP_SQPOLL : 0;
    if (io_uring_queue_init(1, &ring_io, flags) != 0) {
        ++g_expert_pack.iouring_fallbacks;
        return false;
    }

    io_uring_sqe *sqe = io_uring_get_sqe(&ring_io);
    if (!sqe) {
        io_uring_queue_exit(&ring_io);
        ++g_expert_pack.iouring_fallbacks;
        return false;
    }

    io_uring_prep_read(sqe, source->fd_direct, dst, (unsigned)read_sz, (off_t)entry->offset);
    io_uring_sqe_set_data64(sqe, 1);

    const auto submit_start = std::chrono::steady_clock::now();
    const int submit_rc = io_uring_submit(&ring_io);
    const auto submit_end = std::chrono::steady_clock::now();
    g_expert_pack.iouring_submit_us.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count());
    if (submit_rc < 0) {
        io_uring_queue_exit(&ring_io);
        ++g_expert_pack.iouring_fallbacks;
        return false;
    }

    io_uring_cqe *cqe = nullptr;
    const auto wait_start = std::chrono::steady_clock::now();
    const int wait_rc = io_uring_wait_cqe(&ring_io, &cqe);
    const auto wait_end = std::chrono::steady_clock::now();
    g_expert_pack.iouring_wait_us.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count());
    bool ok = false;
    if (wait_rc == 0 && cqe && cqe->res == (int)read_sz) {
        ok = true;
    }
    if (cqe) io_uring_cqe_seen(&ring_io, cqe);
    io_uring_queue_exit(&ring_io);

    if (ok) {
        ++g_expert_pack.iouring_reads;
        g_expert_pack.iouring_bytes.fetch_add(sz);
        return true;
    }

    ++g_expert_pack.iouring_fallbacks;
    return false;
#else
    (void)entry;
    (void)dst;
    (void)sz;
    return false;
#endif
}

static bool expert_pack_read_entry(const expert_pack_entry *entry, void *dst, size_t sz) {
    expert_pack_source *source = expert_pack_source_for_entry(entry);
    if (!entry || !source || !source->file || entry->nbytes != sz) return false;

    std::call_once(g_ram_tier_once, ram_tier_init);

#if !defined(_WIN32)
    if (g_expert_pack.io_backend == 2 && source->fd_direct >= 0 &&
            expert_pack_env_bool("GGML_MOE_IO_URING_SINGLE", false)) {
        if (expert_pack_read_entry_iouring(entry, dst, sz)) {
            return true;
        }
    }
    if ((g_expert_pack.io_backend == 1 || g_expert_pack.io_backend == 2) && source->fd_direct >= 0) {
        const uint64_t alignment = expert_pack_direct_alignment();
        const size_t read_sz = (size_t)align_up_u64((uint64_t)sz, alignment);
        if ((entry->offset % alignment) == 0 &&
                ((uintptr_t)dst % alignment) == 0 &&
                read_sz >= sz) {
            char *out = (char *)dst;
            size_t done = 0;
            while (done < read_sz) {
                const size_t chunk = std::min(read_sz - done, (size_t)64 * 1024 * 1024);
                const ssize_t got = ::pread(source->fd_direct, out + done, chunk, (off_t)(entry->offset + done));
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
    if (_fseeki64(source->file, (int64_t)entry->offset, SEEK_SET) != 0) {
#else
    if (::fseeko(source->file, (off_t)entry->offset, SEEK_SET) != 0) {
#endif
        ++g_expert_pack.read_failures;
        return false;
    }
    if (!expert_pack_read_exact(source->file, dst, sz)) {
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

static bool copy_profile_enabled() {
    const char *env = std::getenv("GGML_MOE_COPY_PROFILE_OUT");
    return env && env[0];
}

static void copy_profile_record(
        const char *op,
        const char *tensor,
        int expert_idx,
        size_t bytes,
        bool pack_hit,
        bool ram_hit,
        bool iouring,
        double slot_wait_ms,
        double host_ms,
        double io_wait_ms,
        double enqueue_ms,
        double h2d_ms,
        double wall_ms) {
    const char *path = std::getenv("GGML_MOE_COPY_PROFILE_OUT");
    if (!path || !path[0]) return;

    static std::mutex mu;
    static bool header_written = false;
    static uint64_t seq = 0;
    std::lock_guard<std::mutex> lk(mu);

    FILE *f = std::fopen(path, "a");
    if (!f) return;
    if (!header_written) {
        std::fprintf(f,
                "seq,op,tensor,expert_idx,bytes,pack_hit,ram_hit,iouring,slot_wait_ms,host_ms,io_wait_ms,enqueue_ms,h2d_ms,wall_ms\n");
        header_written = true;
    }
    std::fprintf(f,
            "%lu,%s,%s,%d,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned long)++seq,
            op ? op : "",
            tensor ? tensor : "",
            expert_idx,
            bytes,
            pack_hit ? 1 : 0,
            ram_hit ? 1 : 0,
            iouring ? 1 : 0,
            slot_wait_ms,
            host_ms,
            io_wait_ms,
            enqueue_ms,
            h2d_ms,
            wall_ms);
    std::fclose(f);
}

static bool expert_pack_ram_tier_copy_h2d(
        const expert_pack_entry *pack_entry,
        void *dst,
        size_t sz,
        cudaStream_t st,
        batch_copy_trace *copy_trace = nullptr) {
    if (!pack_entry) return false;
    std::call_once(g_ram_tier_once, ram_tier_init);
    if (g_expert_pack.ram_tier_base == nullptr) return false;

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
    return false;
}

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
#if defined(GGML_MOE_HAS_LIBURING) && !defined(_WIN32)
    if (ring.uring) {
        io_uring_queue_exit(ring.uring);
        delete ring.uring;
        ring.uring = nullptr;
        ring.uring_depth = 0;
    }
#endif
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
        if (ring.iouring_batches > 0) {
            const double inflight_avg = ring.iouring_inflight_samples > 0 ?
                (double)ring.iouring_inflight_sum / (double)ring.iouring_inflight_samples : 0.0;
            std::fprintf(stderr,
                "[moe_stream_batch] pinned staging%s iouring: batches=%lu jobs=%lu submit_calls=%lu wait_calls=%lu cqes=%lu "
                "inflight_avg=%.2f inflight_max=%lu batch_hist=1:%lu,2-4:%lu,5-8:%lu,9-16:%lu,17-32:%lu,gt32:%lu\n",
                name, ring.iouring_batches, ring.iouring_jobs, ring.iouring_submit_calls,
                ring.iouring_wait_calls, ring.iouring_cqes, inflight_avg, ring.iouring_inflight_max,
                ring.iouring_batch_hist[0], ring.iouring_batch_hist[1], ring.iouring_batch_hist[2],
                ring.iouring_batch_hist[3], ring.iouring_batch_hist[4], ring.iouring_batch_hist[5]);
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
        batch_copy_trace *copy_trace = nullptr,
        const char *trace_op = nullptr,
        const char *tensor_name = nullptr,
        int expert_idx = -1) {
    if (copy_trace) {
        copy_trace->pack_hit = pack_entry != nullptr;
        copy_trace->ram_hit = false;
    }
    const bool profile_copy = copy_profile_enabled() && trace_op && trace_op[0];
    const auto copy_profile_start = profile_copy ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // RAM tier fast path: if the entry is resident in the registered mmap,
    // do a direct H2D from the pinned region, bypassing the staging slot.
    if (expert_pack_ram_tier_copy_h2d(pack_entry, dst, sz, st, copy_trace)) {
        if (profile_copy) {
            const auto copy_profile_end = std::chrono::steady_clock::now();
            copy_profile_record(
                    trace_op, tensor_name, expert_idx, sz,
                    pack_entry != nullptr, true, false,
                    0.0, 0.0, 0.0, 0.0, -1.0,
                    std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
        }
        return true;
    }

    const bool use_pinned_stage = stage_pinned_enabled() || pack_entry;
    if (use_pinned_stage) {
        if (pinned_stage_ensure(ring, sz, pack_entry != nullptr)) {
            const bool profile_stage = pinned_stage_profile_enabled();
            pinned_stage_slot &slot = ring.slots[ring.next++ % ring.slots.size()];
            double slot_wait_ms = 0.0;
            if (slot.pending) {
                const auto wait_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto copy_wait_start = profile_copy && !profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (cudaEventSynchronize(slot.done) != cudaSuccess) return false;
                if (profile_stage) {
                    const auto wait_end = std::chrono::steady_clock::now();
                    slot_wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
                    ring.slot_wait_ms += slot_wait_ms;
                } else if (profile_copy) {
                    const auto wait_end = std::chrono::steady_clock::now();
                    slot_wait_ms = std::chrono::duration<double, std::milli>(wait_end - copy_wait_start).count();
                }
                slot.pending = false;
                ++ring.waits;
            }
            pinned_stage_collect_timing(ring, slot);

            const bool measure_host = profile_stage || profile_copy;
            const auto host_start = measure_host ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (pack_entry) {
                if (!expert_pack_read_entry(pack_entry, slot.host, sz)) {
                    return false;
                }
            } else {
                std::memcpy(slot.host, host_data, sz);
            }
            double host_ms = 0.0;
            if (profile_stage) {
                const auto host_end = std::chrono::steady_clock::now();
                host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
                ring.host_stage_ms += host_ms;
            } else if (profile_copy) {
                const auto host_end = std::chrono::steady_clock::now();
                host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
            }
            const bool measure_enqueue = profile_stage || profile_copy;
            const auto enqueue_start = measure_enqueue ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
            double enqueue_ms = 0.0;
            if (profile_stage) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
                ring.enqueue_ms += enqueue_ms;
            } else if (profile_copy) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
            }
            slot.pending = true;
            ++ring.copies;
            if (profile_copy) {
                const auto copy_profile_end = std::chrono::steady_clock::now();
                copy_profile_record(
                        trace_op, tensor_name, expert_idx, sz,
                        pack_entry != nullptr, false, false,
                        slot_wait_ms, host_ms, 0.0, enqueue_ms, -1.0,
                        std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
            }
            return true;
        }
        ++ring.fallbacks;
    }

    const auto enqueue_start = profile_copy ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool ok = cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (profile_copy && ok) {
        const auto copy_profile_end = std::chrono::steady_clock::now();
        copy_profile_record(
                trace_op, tensor_name, expert_idx, sz,
                pack_entry != nullptr, false, false,
                0.0, 0.0, 0.0,
                std::chrono::duration<double, std::milli>(copy_profile_end - enqueue_start).count(),
                -1.0,
                std::chrono::duration<double, std::milli>(copy_profile_end - copy_profile_start).count());
    }
    return ok;
}

static bool batch_cache_copy_h2d(
        void *dst, const void *host_data, size_t sz, cudaStream_t st,
        const expert_pack_entry *pack_entry) {
    return batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, nullptr);
}

template <typename Job>
static bool expert_pack_iouring_copy_jobs(
        const std::vector<Job> &jobs,
        size_t expert_bytes,
        cudaStream_t st,
        pinned_stage_ring &ring,
        const char *trace_op) {
#if defined(GGML_MOE_HAS_LIBURING) && !defined(_WIN32)
    if (jobs.empty()) return true;
    if (g_expert_pack.io_backend != 2) return false;
    if (!pinned_stage_ensure(ring, expert_bytes, true)) return false;

    const size_t alignment = expert_pack_direct_alignment();
    const size_t read_sz = (size_t)align_up_u64((uint64_t)expert_bytes, (uint64_t)alignment);
    const size_t depth = std::min(expert_pack_io_depth(), ring.slots.size());
    if (depth == 0 || read_sz == 0) return false;

    std::vector<size_t> read_jobs;
    read_jobs.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Job &job = jobs[i];
        if (!job.pack_entry ||
                job.pack_entry->nbytes != expert_bytes ||
                (job.pack_entry->offset % alignment) != 0) {
            return false;
        }
        const expert_pack_source *source = expert_pack_source_for_entry(job.pack_entry);
        if (!source || source->fd_direct < 0) {
            return false;
        }
        batch_copy_trace copy_trace;
        copy_trace.pack_hit = true;
        const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (expert_pack_ram_tier_copy_h2d(job.pack_entry, job.dst, expert_bytes, st, &copy_trace)) {
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    trace_op,
                    job.tensor,
                    job.expert_idx,
                    expert_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
            }
            continue;
        }
        if (host_prefetch_copy_h2d(job.pack_entry, job.tensor, job.expert_idx, job.dst, expert_bytes, st)) {
            copy_trace.pack_hit = true;
            copy_trace.ram_hit = false;
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    trace_op,
                    job.tensor,
                    job.expert_idx,
                    expert_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
            }
            continue;
        }
        read_jobs.push_back(i);
    }
    if (read_jobs.empty()) {
        return true;
    }
    expert_pack_record_iouring_batch(read_jobs.size());
    ring.iouring_batches += 1;
    ring.iouring_jobs += read_jobs.size();
    const size_t bucket = expert_pack_iouring_batch_bucket(read_jobs.size());
    if (bucket < 6) {
        ring.iouring_batch_hist[bucket] += 1;
    }

    struct pending_job {
        size_t job_idx = 0;
        size_t slot_idx = 0;
        size_t bytes = 0;
        std::chrono::steady_clock::time_point copy_start;
    };

    std::vector<size_t> job_order;
    const bool sort_by_offset = expert_pack_env_bool("GGML_MOE_IO_SORT_OFFSET", false);
    if (sort_by_offset && read_jobs.size() > 1) {
        job_order.resize(read_jobs.size());
        for (size_t i = 0; i < read_jobs.size(); ++i) {
            job_order[i] = i;
        }
        std::stable_sort(job_order.begin(), job_order.end(),
            [&](size_t a, size_t b) {
                const expert_pack_entry *ea = jobs[read_jobs[a]].pack_entry;
                const expert_pack_entry *eb = jobs[read_jobs[b]].pack_entry;
                if (ea->source_idx != eb->source_idx) return ea->source_idx < eb->source_idx;
                return ea->offset < eb->offset;
            });
    }
    auto ordered_job_idx = [&](size_t seq_idx) -> size_t {
        const size_t read_idx = job_order.empty() ? seq_idx : job_order[seq_idx];
        return read_jobs[read_idx];
    };

    const unsigned int flags = expert_pack_env_bool("GGML_MOE_IO_SQPOLL", false) ? IORING_SETUP_SQPOLL : 0;
    if (!ring.uring || ring.uring_depth != depth) {
        if (ring.uring) {
            io_uring_queue_exit(ring.uring);
            delete ring.uring;
            ring.uring = nullptr;
            ring.uring_depth = 0;
        }
        ring.uring = new io_uring();
        if (io_uring_queue_init((unsigned)depth, ring.uring, flags) != 0) {
            delete ring.uring;
            ring.uring = nullptr;
            ++g_expert_pack.iouring_fallbacks;
            return false;
        }
        ring.uring_depth = depth;
    }
    io_uring *ring_io = ring.uring;

    std::vector<pending_job> pending(depth);
    std::vector<size_t> free_pending;
    free_pending.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        free_pending.push_back(depth - 1 - i);
    }

    const bool profile_stage = pinned_stage_profile_enabled();
    const bool profile_copy = copy_profile_enabled();
    auto submit_one = [&](size_t job_idx, size_t slot_idx, size_t pending_idx) -> bool {
        const Job &job = jobs[job_idx];
        pinned_stage_slot &slot = ring.slots[slot_idx];
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

        io_uring_sqe *sqe = io_uring_get_sqe(ring_io);
        if (!sqe) return false;
        pending[pending_idx] = {
            job_idx,
            slot_idx,
            read_sz,
            (batch_ttft_trace_enabled() || profile_copy) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}
        };
        const expert_pack_source *source = expert_pack_source_for_entry(job.pack_entry);
        if (!source || source->fd_direct < 0) return false;
        io_uring_prep_read(sqe, source->fd_direct, slot.host, (unsigned)read_sz, (off_t)job.pack_entry->offset);
        io_uring_sqe_set_data64(sqe, (uint64_t)pending_idx + 1);
        return true;
    };

    const auto submit_start = std::chrono::steady_clock::now();
    size_t next_job = 0;
    size_t inflight = 0;
    const size_t refill_batch = std::min(expert_pack_io_refill_batch(), depth);
    struct reusable_pending {
        size_t pending_idx = 0;
        size_t slot_idx = 0;
    };
    std::vector<reusable_pending> reusable_pending_slots;
    reusable_pending_slots.reserve(depth);
    while (next_job < read_jobs.size() && inflight < depth) {
        const size_t pending_idx = free_pending.back();
        free_pending.pop_back();
        const size_t slot_idx = ring.next++ % ring.slots.size();
        if (!submit_one(ordered_job_idx(next_job), slot_idx, pending_idx)) {
            ++g_expert_pack.iouring_fallbacks;
            return false;
        }
        ++next_job;
        ++inflight;
    }
    if (io_uring_submit(ring_io) < 0) {
        ++g_expert_pack.iouring_fallbacks;
        return false;
    }
    ++g_expert_pack.iouring_submit_calls;
    ++ring.iouring_submit_calls;
    const auto submit_end = std::chrono::steady_clock::now();
    g_expert_pack.iouring_submit_us.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count());

    size_t completed = 0;
    while (completed < read_jobs.size()) {
        g_expert_pack.iouring_inflight_sum.fetch_add(inflight);
        ++g_expert_pack.iouring_inflight_samples;
        expert_pack_atomic_max(g_expert_pack.iouring_inflight_max, inflight);
        ring.iouring_inflight_sum += inflight;
        ++ring.iouring_inflight_samples;
        if (ring.iouring_inflight_max < inflight) {
            ring.iouring_inflight_max = inflight;
        }
        auto handle_cqe = [&](io_uring_cqe *cqe) -> bool {
            const uint64_t data = io_uring_cqe_get_data64(cqe);
            const size_t pending_idx = data == 0 ? SIZE_MAX : (size_t)data - 1;
            if (pending_idx >= pending.size() || cqe->res != (int)pending[pending_idx].bytes) {
                io_uring_cqe_seen(ring_io, cqe);
                ++g_expert_pack.iouring_fallbacks;
                return false;
            }

            const pending_job done = pending[pending_idx];
            const Job &job = jobs[done.job_idx];
            pinned_stage_slot &slot = ring.slots[done.slot_idx];
            io_uring_cqe_seen(ring_io, cqe);
            ++g_expert_pack.iouring_cqes;
            ++ring.iouring_cqes;

            const bool measure_enqueue = profile_stage || profile_copy;
            const auto enqueue_start = measure_enqueue ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (profile_stage && slot.copy_start) {
                if (cudaEventRecord(slot.copy_start, st) != cudaSuccess) {
                    return false;
                }
            }
            if (cudaMemcpyAsync(job.dst, slot.host, expert_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) {
                return false;
            }
            if (profile_stage && slot.copy_done) {
                if (cudaEventRecord(slot.copy_done, st) != cudaSuccess) {
                    return false;
                }
                slot.timing_pending = true;
            }
            if (cudaEventRecord(slot.done, st) != cudaSuccess) {
                cudaStreamSynchronize(st);
                return false;
            }
            double enqueue_ms = 0.0;
            if (profile_stage) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
                ring.enqueue_ms += enqueue_ms;
            } else if (profile_copy) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
            }
            slot.pending = true;
            ++ring.copies;
            ++g_expert_pack.iouring_reads;
            g_expert_pack.iouring_bytes.fetch_add(expert_bytes);
            ++g_expert_pack.iouring_h2d_enqueues;

            if (done.copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(copy_end - done.copy_start).count();
                if (batch_ttft_trace_enabled()) {
                    batch_ttft_trace_record(
                        trace_op,
                        job.tensor,
                        job.expert_idx,
                        expert_bytes,
                        false,
                        true,
                        false,
                        wall_ms);
                }
                if (profile_copy) {
                    copy_profile_record(
                            trace_op, job.tensor, job.expert_idx, expert_bytes,
                            true, false, true,
                            0.0, 0.0, wall_ms, enqueue_ms, -1.0, wall_ms);
                }
            }

            ++completed;
            --inflight;
            reusable_pending_slots.push_back({pending_idx, done.slot_idx});
            return true;
        };

        auto refill_pending = [&]() -> bool {
            size_t submitted = 0;
            while (next_job < read_jobs.size() && inflight < depth && !reusable_pending_slots.empty() && submitted < refill_batch) {
                const reusable_pending reusable = reusable_pending_slots.back();
                reusable_pending_slots.pop_back();
                const size_t pending_idx = reusable.pending_idx;
                const size_t slot_idx = reusable.slot_idx;
                if (!submit_one(ordered_job_idx(next_job), slot_idx, pending_idx)) {
                    ++g_expert_pack.iouring_fallbacks;
                    return false;
                }
                ++next_job;
                ++inflight;
                ++submitted;
            }
            if (submitted > 0) {
                if (io_uring_submit(ring_io) < 0) {
                    ++g_expert_pack.iouring_fallbacks;
                    return false;
                }
                ++g_expert_pack.iouring_submit_calls;
                ++ring.iouring_submit_calls;
            }
            return true;
        };

        const auto wait_start = std::chrono::steady_clock::now();
        io_uring_cqe *cqe = nullptr;
        ++g_expert_pack.iouring_wait_calls;
        ++ring.iouring_wait_calls;
        const int wait_rc = io_uring_wait_cqe(ring_io, &cqe);
        const auto wait_end = std::chrono::steady_clock::now();
        g_expert_pack.iouring_wait_us.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count());
        if (wait_rc != 0 || !cqe) {
            ++g_expert_pack.iouring_fallbacks;
            return false;
        }
        if (!handle_cqe(cqe)) return false;
        if (refill_batch == 1 && !refill_pending()) return false;

        while (completed < read_jobs.size() && inflight > 0) {
            io_uring_cqe *extra_cqe = nullptr;
            const int peek_rc = io_uring_peek_cqe(ring_io, &extra_cqe);
            if (peek_rc != 0 || !extra_cqe) {
                break;
            }
            if (!handle_cqe(extra_cqe)) return false;
            if (refill_batch == 1 && !refill_pending()) return false;
        }
        if (refill_batch > 1 && !refill_pending()) {
            return false;
        }
    }

    return true;
#else
    (void)jobs;
    (void)expert_bytes;
    (void)st;
    (void)ring;
    (void)trace_op;
    return false;
#endif
}

static int batch_cache_insert_slot(
        batch_vram_cache *c, uintptr_t key, const void *host_data, size_t sz, cudaStream_t st,
        bool allow_evict, bool preload, const int *avoid_slots = nullptr, int n_avoid_slots = 0,
        bool do_copy = true, const char *tensor_name = nullptr, int expert_idx = -1,
        bool prefetch_down = false, bool pin_preload = true, bool async_prefetch = false,
        uint64_t profile_count = 0) {
    if (!c || !c->pool || c->n_slots == 0 || sz > c->slot_sz) return -1;
    const bool pin_slot = preload && pin_preload && profile_protect_enabled();
    if (pin_slot && c->pinned >= profile_preload_slot_budget(c)) return -1;

    int slot = -1;
    uint64_t oldest = UINT64_MAX;
    uint32_t lowest_hits = UINT32_MAX;
    uint32_t lowest_profile_count = UINT32_MAX;
    const bool lfu_lru = cache_policy_lfu_lru_enabled();
    const bool profile_lfu_lru = cache_policy_profile_lfu_lru_enabled();
    const bool hybrid_profile_lfu_lru = cache_policy_hybrid_profile_lfu_lru_enabled();
    const bool use_profile_score =
        profile_lfu_lru || (hybrid_profile_lfu_lru && c->clock >= cache_policy_hybrid_after());
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && !g_cache_policy_diag_registered.exchange(true)) {
        std::atexit(cache_policy_diag_report_atexit);
    }
    for (int i = 0; i < c->n_slots; ++i) {
        if (batch_slot_is_avoided(i, avoid_slots, n_avoid_slots)) continue;
        if (c->slot_key[i] == 0) {
            slot = i;
            break;
        }
        if (!allow_evict) continue;
        if (c->slot_pinned[i]) continue;
        if (use_profile_score) {
            if (c->slot_profile_count[i] < lowest_profile_count ||
                    (c->slot_profile_count[i] == lowest_profile_count &&
                     (c->slot_hits[i] < lowest_hits ||
                      (c->slot_hits[i] == lowest_hits && c->slot_used[i] < oldest)))) {
                lowest_profile_count = c->slot_profile_count[i];
                lowest_hits = c->slot_hits[i];
                oldest = c->slot_used[i];
                slot = i;
            }
        } else if (lfu_lru) {
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
    if (!batch_cache_wait_slot_ready(c, slot)) return -1;
    if (c->slot_pinned[slot] && !pin_slot && c->pinned > 0) {
        --c->pinned;
    }
    if (c->slot_key[slot] != 0 && c->slot_prefetch_down[slot]) {
        ++c->down_prefetch_evicted;
    }
    if (profile_count == 0 && (profile_lfu_lru || hybrid_profile_lfu_lru)) {
        profile_count = profile_count_for_key(key);
    }
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && c->slot_key[slot] != 0) {
        ++g_cache_policy_diag.profile_policy_evictions;
        if (c->slot_profile_count[slot] > 0) {
            ++g_cache_policy_diag.profile_policy_victim_count_nonzero;
            g_cache_policy_diag.profile_policy_victim_count_sum.fetch_add(c->slot_profile_count[slot]);
        }
    }
    c->slot_key[slot] = key;
    c->slot_used[slot] = c->clock++;
    c->slot_hits[slot] = 0;
    c->slot_profile_count[slot] = profile_count > UINT32_MAX ? UINT32_MAX : (uint32_t)profile_count;
    if ((profile_lfu_lru || hybrid_profile_lfu_lru) && c->slot_profile_count[slot] > 0) {
        ++g_cache_policy_diag.inserted_profile_count_nonzero;
        g_cache_policy_diag.inserted_profile_count_sum.fetch_add(c->slot_profile_count[slot]);
    }
    if (pin_slot && !c->slot_pinned[slot]) {
        ++c->pinned;
    }
    c->slot_pinned[slot] = pin_slot;
    if (pin_slot) {
        profile_pinned_key_record((uint64_t)key);
    }
    c->slot_prefetch_down[slot] = prefetch_down;
    c->slot_pending[slot] = false;
    void *dst = (char *)c->pool + (size_t)slot * c->slot_sz;
    auto clear_slot = [&]() {
        c->slot_key[slot] = 0;
        c->slot_used[slot] = 0;
        c->slot_hits[slot] = 0;
        c->slot_profile_count[slot] = 0;
        if (c->slot_pinned[slot] && c->pinned > 0) {
            --c->pinned;
        }
        c->slot_pinned[slot] = false;
        c->slot_prefetch_down[slot] = false;
        c->slot_pending[slot] = false;
    };
    if (do_copy) {
        const expert_pack_entry *pack_entry = expert_pack_lookup(tensor_name, expert_idx, sz);
        batch_copy_trace copy_trace;
        const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const char *copy_op = preload ? "preload_load" : "runtime_load";
        if (!batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, &copy_trace,
                    copy_op, tensor_name, expert_idx)) {
            if (pack_entry && !batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, nullptr, &copy_trace,
                        copy_op, tensor_name, expert_idx)) {
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
        if (async_prefetch) {
            if (!c->slot_ready[slot] &&
                    cudaEventCreateWithFlags(&c->slot_ready[slot], cudaEventDisableTiming) != cudaSuccess) {
                clear_slot();
                return -1;
            }
            if (cudaEventRecord(c->slot_ready[slot], st) != cudaSuccess) {
                clear_slot();
                return -1;
            }
            c->slot_pending[slot] = true;
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
        const int existing_slot = batch_cache_find_slot(cache, key);
        if (existing_slot >= 0) {
            if (pin_preload && profile_protect_enabled()) {
                if (batch_cache_promote_profile_slot(cache, existing_slot, e.count)) {
                    profile_pinned_key_record((uint64_t)key);
                }
            }
            continue;
        }
        const char *expert_host = (const char *)src0_data + (size_t)e.expert_idx * nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, src0_bytes, st, allow_evict, true,
                nullptr, 0, true, tensor_name, e.expert_idx, false, pin_preload, false, e.count) < 0) break;
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
    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return;
    if (profile_protect_enabled() && cache->pinned < profile_preload_slot_budget(cache)) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    preload_profile_entries_for_tensor(
        g_prompt_profile, "prompt profile", tensor_name, src0_data, n_as, nb02, src0_bytes, st,
        false, false, false, false, prompt_profile_preload_scan_budget(cache));
}

static bool registered_tensor_lookup(const char *tensor_name, registered_tensor *out) {
    if (!tensor_name || !tensor_name[0] || !out) return false;
    std::lock_guard<std::mutex> lk(g_registered_mu);
    for (const registered_tensor &t : g_registered_tensors) {
        if (std::strcmp(t.name, tensor_name) == 0) {
            *out = t;
            return true;
        }
    }
    return false;
}

static bool trace_entry_matches(const batch_route_trace_entry &e, const char *tensor_name, int expert_idx, size_t expert_bytes) {
    return e.expert_idx == expert_idx &&
        e.expert_bytes == expert_bytes &&
        tensor_name && std::strcmp(e.tensor, tensor_name) == 0;
}

static void trace_prefetch_on_hit(const char *tensor_name, int expert_idx, size_t expert_bytes) {
    trace_prefetch_init_once();
    if (!g_trace_prefetch.enabled || !g_batch.prefetch_stream || !tensor_name || !tensor_name[0]) return;

    std::lock_guard<std::mutex> lk(g_trace_prefetch.mu);
    ++g_trace_prefetch.calls;

    if (g_trace_prefetch.cursor < g_trace_prefetch.trace.size() &&
            trace_entry_matches(g_trace_prefetch.trace[g_trace_prefetch.cursor], tensor_name, expert_idx, expert_bytes)) {
        ++g_trace_prefetch.cursor;
        ++g_trace_prefetch.matched;
    } else {
        const size_t scan_end = std::min(g_trace_prefetch.trace.size(), g_trace_prefetch.cursor + g_trace_prefetch.window);
        size_t found = SIZE_MAX;
        for (size_t i = g_trace_prefetch.cursor; i < scan_end; ++i) {
            if (trace_entry_matches(g_trace_prefetch.trace[i], tensor_name, expert_idx, expert_bytes)) {
                found = i;
                break;
            }
        }
        if (found != SIZE_MAX) {
            g_trace_prefetch.cursor = found + 1;
            ++g_trace_prefetch.matched;
            ++g_trace_prefetch.resync;
        }
    }

    const size_t min_prefetch_cursor = std::min(
        g_trace_prefetch.trace.size(),
        g_trace_prefetch.cursor + g_trace_prefetch.lead_events);
    if (g_trace_prefetch.prefetch_cursor < min_prefetch_cursor) {
        g_trace_prefetch.prefetch_cursor = min_prefetch_cursor;
    }

    struct trace_prefetch_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
        int expert_idx = -1;
        char tensor[128] = {};
    };
    std::vector<trace_prefetch_job> jobs;
    size_t job_bytes = 0;

    int examined = 0;
    const size_t end = std::min(
        g_trace_prefetch.trace.size(),
        g_trace_prefetch.cursor + g_trace_prefetch.lead_events + g_trace_prefetch.window);
    for (size_t i = g_trace_prefetch.prefetch_cursor; i < end && examined < g_trace_prefetch.max_loads; ++i, ++examined) {
        g_trace_prefetch.prefetch_cursor = i + 1;
        const batch_route_trace_entry &e = g_trace_prefetch.trace[i];
        if (job_bytes != 0 && e.expert_bytes != job_bytes) {
            continue;
        }
        const expert_pack_entry *pack_entry = expert_pack_lookup(e.tensor, e.expert_idx, e.expert_bytes);
        registered_tensor rt;
        const bool have_registered = registered_tensor_lookup(e.tensor, &rt);
        const bool use_registered = !pack_entry && have_registered &&
            rt.data && e.expert_idx >= 0 && e.expert_idx < rt.n_as && rt.expert_bytes == e.expert_bytes;
        if (!pack_entry && !use_registered) {
            ++g_trace_prefetch.missing_tensor;
            continue;
        }
        batch_vram_cache *cache = batch_cache_get(e.expert_bytes);
        if (!cache) {
            ++g_trace_prefetch.cache_unavailable;
            continue;
        }
        const uintptr_t key = batch_key_hash(e.tensor, e.expert_idx);
        if (batch_cache_find_slot(cache, key) >= 0) {
            ++g_trace_prefetch.cached;
            continue;
        }
        const char *expert_host = use_registered ? (const char *)rt.data + (size_t)e.expert_idx * rt.nb02 : nullptr;
        const int slot = batch_cache_insert_slot(cache, key, expert_host, e.expert_bytes, g_batch.prefetch_stream,
                false, true, nullptr, 0, false, e.tensor, e.expert_idx, false, false, false);
        if (slot >= 0) {
            void *dst = (char *)cache->pool + (size_t)slot * cache->slot_sz;
            trace_prefetch_job job;
            job.slot = slot;
            job.dst = dst;
            job.host_data = expert_host;
            job.pack_entry = pack_entry;
            job.expert_idx = e.expert_idx;
            std::snprintf(job.tensor, sizeof(job.tensor), "%s", e.tensor);
            jobs.push_back(job);
            job_bytes = e.expert_bytes;
        }
    }

    if (jobs.empty()) return;

    auto mark_jobs_pending = [&](batch_vram_cache *cache, const std::vector<trace_prefetch_job> &pending_jobs) -> bool {
        for (const trace_prefetch_job &job : pending_jobs) {
            if (!cache->slot_ready[job.slot] &&
                    cudaEventCreateWithFlags(&cache->slot_ready[job.slot], cudaEventDisableTiming) != cudaSuccess) {
                return false;
            }
            if (cudaEventRecord(cache->slot_ready[job.slot], g_batch.prefetch_stream) != cudaSuccess) {
                return false;
            }
            cache->slot_pending[job.slot] = true;
        }
        return true;
    };

    batch_vram_cache *cache = batch_cache_get(job_bytes);
    if (!cache) {
        g_trace_prefetch.cache_unavailable += jobs.size();
        return;
    }

    bool copied = expert_pack_iouring_copy_jobs(jobs, job_bytes, g_batch.prefetch_stream, g_batch.stage_ring, "trace_prefetch");
    if (!copied) {
        copied = true;
        for (const trace_prefetch_job &job : jobs) {
            batch_copy_trace copy_trace;
            if (!batch_cache_copy_h2d(g_batch.stage_ring, job.dst, job.host_data, job_bytes,
                    g_batch.prefetch_stream, job.pack_entry, &copy_trace,
                    "trace_prefetch", job.tensor, job.expert_idx)) {
                copied = false;
                break;
            }
        }
    }
    if (!copied || !mark_jobs_pending(cache, jobs)) {
        for (const trace_prefetch_job &job : jobs) {
            batch_cache_clear_slot(cache, job.slot);
        }
        return;
    }
    g_trace_prefetch.loads += jobs.size();
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

static __device__ __forceinline__ int moe_q8k_nearest_even_int(float fval) {
    return __float2int_rn(fval);
}

static __global__ void moe_quantize_row_q8_k_kernel(const float * x, block_q8_K * y, int nblocks) {
    const int ib = blockIdx.x;
    if (ib >= nblocks || threadIdx.x != 0) {
        return;
    }

    const float * xb = x + (size_t)ib * QK_K;
    block_q8_K * yb = y + ib;

    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
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

    const float d = amax / 127.0f;
    const float id = 127.0f / amax;
    for (int j = 0; j < QK_K; ++j) {
        int v = moe_q8k_nearest_even_int(id * xb[j]);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
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
    yb->d = d;
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

static __device__ __forceinline__ int moe_q3_k_scale(const block_q3_K * x, int is) {
    const uint8_t * sc = x->scales;
    if (is < 4) {
        return (int)((sc[is] & 0x0f) | (((sc[is + 8] >> 0) & 3) << 4)) - 32;
    }
    if (is < 8) {
        return (int)((sc[is] & 0x0f) | (((sc[is + 4] >> 2) & 3) << 4)) - 32;
    }
    if (is < 12) {
        return (int)((sc[is - 8] >> 4) | (((sc[is] >> 4) & 3) << 4)) - 32;
    }
    return (int)((sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3) << 4)) - 32;
}

static __device__ __forceinline__ int moe_q3_k_q8k_block_sum(
        const block_q3_K * x,
        const block_q8_K * y) {
    int bsum = 0;
    for (int n = 0; n < QK_K/128; ++n) {
        for (int j = 0; j < 4; ++j) {
            const int shift = 2*j;
            const int m = 1 << (4*n + j);
            for (int is0 = 0; is0 < 2; ++is0) {
                const int is = 8*n + 2*j + is0;
                const int sc = moe_q3_k_scale(x, is);
                const int qidx_base = 32*n + 16*is0;
                const int yidx_base = 128*n + 32*j + 16*is0;
                for (int l = 0; l < 16; ++l) {
                    const int q = (int)((x->qs[qidx_base + l] >> shift) & 3) - ((x->hmask[16*is0 + l] & m) ? 0 : 4);
                    bsum += sc * q * (int)y->qs[yidx_base + l];
                }
            }
        }
    }
    return bsum;
}

static __device__ __forceinline__ int moe_sat_i16(int v) {
    return v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
}

static __device__ __forceinline__ int moe_iq4_xs_q8k_maddubs_corrected_sum32(
        const uint8_t * q4,
        const int8_t * y) {
    int sum = 0;
    for (int j = 0; j < 16; j += 2) {
        const int v0 = ((int)kvalues_iq4nl[q4[j + 0] & 0x0f] + 128) * (int)y[j + 0];
        const int v1 = ((int)kvalues_iq4nl[q4[j + 1] & 0x0f] + 128) * (int)y[j + 1];
        sum += moe_sat_i16(v0 + v1) - 128 * ((int)y[j + 0] + (int)y[j + 1]);
    }
    for (int j = 0; j < 16; j += 2) {
        const int v0 = ((int)kvalues_iq4nl[q4[j + 0] >> 4] + 128) * (int)y[16 + j + 0];
        const int v1 = ((int)kvalues_iq4nl[q4[j + 1] >> 4] + 128) * (int)y[16 + j + 1];
        sum += moe_sat_i16(v0 + v1) - 128 * ((int)y[16 + j + 0] + (int)y[16 + j + 1]);
    }
    return sum;
}

static __device__ __forceinline__ int moe_iq4_xs_q8k_block_sum(
        const block_iq4_xs * x,
        const block_q8_K * y) {
    int bsum = 0;
    uint16_t h = x->scales_h;
    for (int ib = 0; ib < QK_K/64; ++ib) {
        const int sc1 = ((x->scales_l[ib] & 0x0f) | ((h << 4) & 0x30)) - 32;
        const int sc2 = ((x->scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
        const int sumi1 = moe_iq4_xs_q8k_maddubs_corrected_sum32(x->qs + 32*ib,      y->qs + 64*ib);
        const int sumi2 = moe_iq4_xs_q8k_maddubs_corrected_sum32(x->qs + 32*ib + 16, y->qs + 64*ib + 32);
        bsum += sc1 * sumi1 + sc2 * sumi2;
        h >>= 4;
    }
    return bsum;
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
        } else if (src0_type == GGML_TYPE_Q3_K) {
            const block_q3_K * x_q3 = (const block_q3_K *)x_row_base;
            const float d = __half2float(x_q3[ib].d) * y_row[ib].d;
            sum += d * (float)moe_q3_k_q8k_block_sum(x_q3 + ib, y_row + ib);
        } else if (src0_type == GGML_TYPE_IQ4_XS) {
            const block_iq4_xs * x_iq4 = (const block_iq4_xs *)x_row_base;
            const float d = __half2float(x_iq4[ib].d) * y_row[ib].d;
            sum += d * (float)moe_iq4_xs_q8k_block_sum(x_iq4 + ib, y_row + ib);
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
    std::fprintf(stderr, "[moe_stream_batch] IQ2_S Q8_K_R8 selftest unavailable in vendor correctness path\n");
    return false;
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
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
        type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ4_XS;
}

static bool moe_tensor_layer_in_simple_range(const char *name, const char *range) {
    if (!range || !range[0]) return true;
    if (!name) return false;
    const char *p = std::strstr(name, "blk.");
    if (!p) return false;
    p += 4;
    char *end = nullptr;
    const long layer = std::strtol(p, &end, 10);
    if (end == p) return false;
    const char *r = range;
    while (*r) {
        char *next = nullptr;
        const long lo = std::strtol(r, &next, 10);
        if (next == r) break;
        long hi = lo;
        if (*next == '-') {
            r = next + 1;
            hi = std::strtol(r, &next, 10);
            if (next == r) return false;
        }
        if (layer >= lo && layer <= hi) return true;
        r = (*next == ',') ? next + 1 : next;
    }
    return false;
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
        const char *layer_prof_env = std::getenv("GGML_MOE_UP_GATE_LAYER_PROFILE");
        g_up_gate_layer_profile_enabled = layer_prof_env && layer_prof_env[0] && layer_prof_env[0] != '0';
        if (g_up_gate_layer_profile_enabled) {
            std::fprintf(stderr, "[moe_stream_batch] up/gate layer profile enabled\n");
        }
        g_bprof.enabled = prof_env && prof_env[0] && prof_env[0] != '0';
        g_uprof.enabled = g_bprof.enabled || up_gate_profile_csv_enabled() || g_up_gate_layer_profile_enabled;
        if (g_batch.stream && (g_bprof.enabled || g_uprof.enabled)) {
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
            if (g_bprof.enabled) {
                std::atexit(batch_profile_report_atexit);
                std::atexit(current_down_overlap_report_atexit);
            }
            if (g_uprof.enabled) {
                std::atexit(up_gate_profile_report_atexit);
            }
            if (g_up_gate_layer_profile_enabled) {
                std::atexit(up_gate_layer_profile_report_atexit);
            }
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
    preload_profile_for_tensor(src0_name, src0_data, n_as, nb02, expert_bytes, g_batch.stream);
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

extern "C" bool ggml_cuda_moe_stream_preload_expert_from_pack_async(
    int src0_type_int,
    const char *src0_name,
    int64_t n_as,
    size_t expert_bytes,
    int expert_idx) {
    if (!init_batch_once()) return false;
    if (!moe_stream_type_supported((ggml_type)src0_type_int) || !src0_name || !src0_name[0]) return false;
    if (expert_idx < 0 || expert_idx >= n_as) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    batch_vram_cache *cache = batch_cache_get(expert_bytes);
    if (!cache) return false;
    const uintptr_t key = batch_key_hash(src0_name, expert_idx);
    if (batch_cache_find_slot(cache, key) >= 0) return true;
    return batch_cache_insert_slot(cache, key, nullptr, expert_bytes, g_batch.stream, false, true,
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
    if (!std::strstr(src0_name, ".ffn_up_exps.") &&
            !std::strstr(src0_name, ".ffn_gate_exps.") &&
            !std::strstr(src0_name, ".ffn_down_exps.")) {
        return true;
    }

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

static bool launch_moe_mmvq_compact_batch(
        ggml_type src0_type,
        const char * d_src0_pool,
        const int32_t * h_x_ids,
        int64_t slot_stride,
        int64_t ne00,
        int64_t ne01,
        const float * d_src1_f32,
        int64_t src1_row_stride,
        const int32_t * h_src1_rows,
        void * d_src1_q8,
        float * d_dst,
        int64_t n_active,
        cudaStream_t st) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
            break;
        default: return false;
    }
    const size_t q8_row_bytes = moe_stream_q8_1_row_bytes(ne00);
    for (int64_t j = 0; j < n_active; ++j) {
        const int32_t slot = h_x_ids[j];
        if (slot < 0) return false;
        const int64_t src1_row = h_src1_rows ? h_src1_rows[j] : j;
        const char * d_expert = d_src0_pool + (size_t)slot * (size_t)slot_stride;
        const float * d_src1_row = d_src1_f32 + (size_t)src1_row * (size_t)src1_row_stride;
        void * d_src1_q8_row = (char *)d_src1_q8 + (size_t)j * q8_row_bytes;
        float * d_dst_row = d_dst + (size_t)j * (size_t)ne01;
        if (!ggml_cuda_moe_stream_mmvq_dev(
                src0_type, d_expert, ne01, ne00, d_src1_row, d_src1_q8_row, d_dst_row, st)) {
            return false;
        }
    }
    return true;
}

static bool launch_moe_mmq_slot_batch(
        ggml_type src0_type,
        const char * d_src0_pool,
        const int * d_src1_q8,
        const int32_t * d_ids_dst,
        const int32_t * d_bounds,
        const int32_t * d_x_ids,
        float * d_dst,
        int64_t ne00,
        int64_t ne01,
        int64_t src0_stride,
        int64_t src0_channel_stride,
        int64_t n_active,
        int64_t dst_cols,
        cudaStream_t st) {
    if (!d_src0_pool || !d_src1_q8 || !d_ids_dst || !d_bounds || !d_x_ids || !d_dst ||
            ne00 <= 0 || ne01 <= 0 || n_active <= 0 || dst_cols <= 0) {
        return false;
    }
    const size_t ts_src0 = ggml_type_size(src0_type);
    if (ts_src0 == 0 || src0_stride % (int64_t)ts_src0 != 0 || src0_channel_stride % (int64_t)ts_src0 != 0) {
        return false;
    }
    const int64_t src0_row_stride_blocks = src0_stride / (int64_t)ts_src0;
    const int64_t src0_channel_stride_blocks = src0_channel_stride / (int64_t)ts_src0;

    const mmq_args args = {
        d_src0_pool, src0_type, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_row_stride_blocks, n_active, ne01,
        n_active, n_active, src0_channel_stride_blocks, 0, 0,
        1, 1, 0, 0, 0,
        false, n_active
    };

    ggml_backend_cuda_context * null_ctx = nullptr;
    switch (src0_type) {
        case GGML_TYPE_IQ3_XXS:
            launch_mul_mat_q<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st);
            break;
        case GGML_TYPE_IQ2_S:
            launch_mul_mat_q<GGML_TYPE_IQ2_S, 8>(*null_ctx, args, st);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q<GGML_TYPE_Q3_K, 8>(*null_ctx, args, st);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q<GGML_TYPE_IQ4_XS, 8>(*null_ctx, args, st);
            break;
        default:
            return false;
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

static bool moe_stream_compare_device_rows(
        const char *tag,
        const float *got_d,
        const float *ref_d,
        void *scratch_host,
        size_t scratch_host_sz,
        int n_rows,
        int64_t ne01,
        cudaStream_t st) {
    const size_t n_vals = (size_t)n_rows * (size_t)ne01;
    const size_t bytes = n_vals * sizeof(float);
    if (!tag || !got_d || !ref_d || !scratch_host || scratch_host_sz < bytes || n_rows <= 0 || ne01 <= 0) {
        return false;
    }

    std::vector<float> got(n_vals);
    if (cudaMemcpyAsync(scratch_host, got_d, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return false;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return false;
    }
    std::memcpy(got.data(), scratch_host, bytes);

    if (cudaMemcpyAsync(scratch_host, ref_d, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return false;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return false;
    }
    const float *ref = (const float *)scratch_host;

    double l1 = 0.0;
    double l2 = 0.0;
    double ref_l2 = 0.0;
    double max_abs = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n_vals; ++i) {
        const double g = (double)got[i];
        const double r = (double)ref[i];
        const double d = g - r;
        const double ad = std::fabs(d);
        l1 += ad;
        l2 += d*d;
        ref_l2 += r*r;
        if (ad > max_abs) {
            max_abs = ad;
        }
        if (!std::isfinite(g) || !std::isfinite(r)) {
            ++bad;
        }
    }

    const double denom = n_vals > 0 ? (double)n_vals : 1.0;
    std::fprintf(stderr,
        "[moe_stream] up/gate MMQ compare %s: rows=%d ne01=%ld l1_avg=%g max=%g rmse=%g rel_rmse=%g bad=%d\n",
        tag, n_rows, (long)ne01, l1 / denom, max_abs,
        std::sqrt(l2 / denom), std::sqrt(l2 / std::max(ref_l2, 1e-30)), bad);
    return true;
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
    int  src0_up_type_int,
    int  src0_gate_type_int,
    const char *src0_up_name,
    const void *src0_up_data,
    const char *src0_gate_name,
    const void *src0_gate_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t up_nb01,
    size_t up_nb02,
    size_t up_expert_bytes,
    size_t gate_nb01,
    size_t gate_nb02,
    size_t gate_expert_bytes,
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
                "[moe_stream] up/gate batch declined: reason=%s tensor=%s rows_stride=%ld up_type=%d gate_type=%d\n",
                reason, src0_up_name ? src0_up_name : "", (long)rows_stride, src0_up_type_int, src0_gate_type_int);
        }
        return false;
    };
    if (!init_batch_once()) return decline("init_batch_once");
    ggml_type src0_type = (ggml_type)src0_up_type_int;
    ggml_type gate_type = (ggml_type)src0_gate_type_int;
    const bool mixed_types = src0_type != gate_type;
    const bool scoped_mixed_iq2_iq3 =
        (src0_type == GGML_TYPE_IQ2_S && gate_type == GGML_TYPE_IQ3_XXS) ||
        (src0_type == GGML_TYPE_IQ3_XXS && gate_type == GGML_TYPE_IQ2_S);
    if (!moe_stream_type_supported(src0_type) || !moe_stream_type_supported(gate_type)) return decline("unsupported_type");
    if (mixed_types && !scoped_mixed_iq2_iq3) return decline("unsupported_mixed_type_pair");
    if (up_expert_bytes == 0 || gate_expert_bytes == 0) return decline("bad_expert_bytes");
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
    ggml_cuda_moe_stream_register_tensor(src0_up_type_int, src0_up_name, src0_up_data, n_as, up_nb02, up_expert_bytes);
    ggml_cuda_moe_stream_register_tensor(src0_gate_type_int, src0_gate_name, src0_gate_data, n_as, gate_nb02, gate_expert_bytes);

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
    const auto wall_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const size_t src0_bytes = up_expert_bytes;
    const size_t nb01 = up_nb01;
    const size_t nb02 = up_nb02;
    const size_t cache_slot_bytes = std::max(up_expert_bytes, gate_expert_bytes);
    batch_ttft_call_scope ttft_scope("call_upgate", src0_up_name, n_active, up_expert_bytes + gate_expert_bytes);
    const size_t src1_f32_bytes = (size_t)n_active * ne00 * sizeof(float);
    const size_t src1_q8_bytes = (size_t)n_active * moe_stream_q8_1_row_bytes(ne00);
    const size_t src1_q8_one_bytes = moe_stream_q8_1_row_bytes(ne00);
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

    batch_vram_cache *cache = batch_cache_get(cache_slot_bytes);
    if (!cache) return decline("cache_unavailable");
    if (!current_down_overlap_enabled()) {
        preload_registered_down_for_active(src0_up_name, active_experts, n_active);
    }

    char up_key_name[128] = {};
    char gate_key_name[128] = {};
    const bool shared_tensor_name = src0_up_name && src0_gate_name && std::strcmp(src0_up_name, src0_gate_name) == 0;
    const bool disambiguate_halves = shared_tensor_name || src0_up_data == src0_gate_data;
    std::snprintf(up_key_name, sizeof(up_key_name), "%s%s", src0_up_name ? src0_up_name : "up", disambiguate_halves ? ":up" : "");
    std::snprintf(gate_key_name, sizeof(gate_key_name), "%s%s", src0_gate_name ? src0_gate_name : "gate", disambiguate_halves ? ":gate" : "");

    const char *profile_upgate_env = std::getenv("GGML_MOE_VRAM_PROFILE_UPGATE");
    const bool profile_upgate = !profile_upgate_env || !profile_upgate_env[0] || profile_upgate_env[0] != '0';
    if (profile_upgate) {
        preload_profile_for_tensor(up_key_name, src0_up_data, n_as, up_nb02, up_expert_bytes, st);
        preload_profile_for_tensor(gate_key_name, src0_gate_data, n_as, gate_nb02, gate_expert_bytes, st);
    }
    if (profile) cudaEventRecord(bc.ev_start, st);

    const char *serial_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_SERIAL");
    const bool serial_up_gate = serial_env && serial_env[0] && serial_env[0] != '0';
    static std::atomic<int> first_serial_up_gate{0};
    if (serial_up_gate && first_serial_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] up/gate serial MMVQ compatibility path active\n");
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
        !mixed_types &&
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
    const char *fused_mmq_env = std::getenv("GGML_MOE_STREAM_UP_GATE_FUSED_MMQ");
    const bool fused_mmq_up_gate =
        fused_mmq_env && fused_mmq_env[0] && fused_mmq_env[0] != '0' &&
        !mixed_types && !prompt_mode && !exact_prompt_q8k && !serial_up_gate &&
        (src0_type == GGML_TYPE_IQ3_XXS || src0_type == GGML_TYPE_IQ2_S);
    static std::atomic<int> first_fused_mmq_up_gate{0};
    if (fused_mmq_up_gate && first_fused_mmq_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] vendor MMQ up/gate path active: type=%d\n", (int)src0_type);
    }
    const char *mmq_compare_env = std::getenv("GGML_MOE_STREAM_UP_GATE_MMQ_COMPARE");
    const bool mmq_compare =
        fused_mmq_up_gate && mmq_compare_env && mmq_compare_env[0] && mmq_compare_env[0] != '0';
    const char *serial_stage_batch_env = std::getenv("GGML_MOE_STREAM_SERIAL_STAGE_BATCH");
    const bool serial_stage_batch =
        serial_stage_batch_env && serial_stage_batch_env[0] && serial_stage_batch_env[0] != '0' &&
        !prompt_mode && !mixed_types && !exact_prompt_q8k;
    static std::atomic<int> first_serial_stage_batch{0};
    if (serial_stage_batch && first_serial_stage_batch.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] serial same-type batched staging active\n");
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
        GGML_UNUSED(serial_up_gate);
        return launch_moe_mmvq_compact_batch(
            src0_type,
            (const char *)cache->pool, h_x_ids, cache->slot_sz,
            ne00, ne01, (const float *)bc.d_src1_f32, ne00, nullptr,
            d_src1_q8, (float *)d_out, n_active, run_stream);
    };

    auto stage_tensor_slots_only = [&](
            const char *key_name, const void *host_base,
            cudaStream_t run_stream, int32_t *d_x_ids,
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
            } else {
                batch_ttft_trace_record("cache_hit", key_name, active_experts[j], src0_bytes, true, false, false, 0.0);
            }
            if (cache_slot < 0) return false;
            h_x_ids[j] = cache_slot;
            if (slots_out) slots_out[j] = cache_slot;
            batch_route_profile_hit(key_name, active_experts[j], src0_bytes);
        }
        return cudaMemcpyAsync(d_x_ids, h_x_ids, ids_bytes, cudaMemcpyHostToDevice, run_stream) == cudaSuccess;
    };

    auto stage_tensor_slots_only_sized = [&](
            const char *key_name, const void *host_base,
            size_t expert_stride, size_t expert_bytes,
            cudaStream_t run_stream, int32_t *d_x_ids,
            int32_t *h_x_ids, int *slots_out,
            const int *avoid_slots, int n_avoid_slots) -> bool {
        for (int j = 0; j < n_active; ++j) {
            const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * expert_stride;
            const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
            int cache_slot = batch_cache_lookup_slot(cache, cache_key);
            if (cache_slot < 0) {
                cache_slot = batch_cache_insert_slot(
                    cache, cache_key, expert_host, expert_bytes, run_stream, true, false,
                    avoid_slots, n_avoid_slots, true, key_name, active_experts[j]);
            } else {
                batch_ttft_trace_record("cache_hit", key_name, active_experts[j], expert_bytes, true, false, false, 0.0);
            }
            if (cache_slot < 0) return false;
            h_x_ids[j] = cache_slot;
            if (slots_out) slots_out[j] = cache_slot;
            batch_route_profile_hit(key_name, active_experts[j], expert_bytes);
        }
        return cudaMemcpyAsync(d_x_ids, h_x_ids, ids_bytes, cudaMemcpyHostToDevice, run_stream) == cudaSuccess;
    };

    struct stage_copy_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
        int expert_idx = -1;
        char tensor[128] = {};
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
        if (expert_pack_iouring_copy_jobs(jobs, src0_bytes, run_stream, ring, "runtime_load")) {
            return cudaGetLastError() == cudaSuccess;
        }
        for (const stage_copy_job &job : jobs) {
            batch_copy_trace copy_trace;
            const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry, &copy_trace,
                        "runtime_load", job.tensor, job.expert_idx)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr, &copy_trace,
                            "runtime_load", job.tensor, job.expert_idx)) {
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

    auto clear_stage_jobs_for_cache = [&](batch_vram_cache *job_cache, const std::vector<stage_copy_job> &jobs) {
        if (!job_cache) return;
        for (const stage_copy_job &job : jobs) {
            batch_cache_clear_slot(job_cache, job.slot);
        }
    };

    std::thread current_down_overlap_thread;
    bool current_down_overlap_ok = true;
    auto join_current_down_overlap = [&]() -> bool {
        if (current_down_overlap_thread.joinable()) {
            current_down_overlap_thread.join();
        }
        return current_down_overlap_ok;
    };

    auto start_current_down_overlap = [&]() {
        if (!current_down_overlap_enabled() || prompt_mode || !bc.prefetch_stream) return;
        ++g_current_down_overlap.calls;

        char down_name[128] = {};
        if (!down_name_for_up_gate(src0_up_name, down_name, sizeof(down_name))) {
            ++g_current_down_overlap.missing_tensor;
            current_down_overlap_tensor_profile_record(src0_up_name, 1, 0, 0, 1, 0, 0);
            return;
        }

        registered_tensor rt;
        bool found = false;
        {
            std::lock_guard<std::mutex> rlk(g_registered_mu);
            for (const registered_tensor &t : g_registered_tensors) {
                if (std::strcmp(t.name, down_name) == 0) {
                    rt = t;
                    found = true;
                    break;
                }
            }
        }
        if (!found || !rt.data || rt.expert_bytes == 0) {
            ++g_current_down_overlap.missing_tensor;
            current_down_overlap_tensor_profile_record(down_name, 1, 0, 0, 1, 0, 0);
            return;
        }

        batch_vram_cache *down_cache = batch_cache_get(rt.expert_bytes);
        if (!down_cache) {
            ++g_current_down_overlap.missing_tensor;
            current_down_overlap_tensor_profile_record(down_name, 1, 0, 0, 1, 0, 0);
            return;
        }

        uint64_t local_cache_hits = 0;
        uint64_t local_missing_pack = 0;
        std::vector<stage_copy_job> down_jobs;
        down_jobs.reserve((size_t)n_active);
        for (int j = 0; j < n_active; ++j) {
            const int expert = active_experts[j];
            if (expert < 0 || expert >= rt.n_as) continue;
            const uintptr_t key = batch_key_hash(rt.name, expert);
            if (batch_cache_find_slot(down_cache, key) >= 0) {
                ++g_current_down_overlap.cache_hits;
                ++local_cache_hits;
                continue;
            }
            const char *expert_host = (const char *)rt.data + (size_t)expert * rt.nb02;
            const int slot = batch_cache_insert_slot(
                down_cache, key, expert_host, rt.expert_bytes, bc.prefetch_stream,
                true, true, nullptr, 0, false, rt.name, expert, true);
            if (slot < 0) continue;

            stage_copy_job job;
            job.slot = slot;
            job.dst = (char *)down_cache->pool + (size_t)slot * down_cache->slot_sz;
            job.host_data = expert_host;
            job.pack_entry = expert_pack_lookup(rt.name, expert, rt.expert_bytes);
            if (!job.pack_entry) {
                ++g_current_down_overlap.missing_pack;
                ++local_missing_pack;
            }
            job.expert_idx = expert;
            std::snprintf(job.tensor, sizeof(job.tensor), "%s", rt.name);
            down_jobs.push_back(job);
        }
        if (down_jobs.empty()) {
            current_down_overlap_tensor_profile_record(rt.name, 1, 0, local_cache_hits, 0, local_missing_pack, 0);
            return;
        }

        current_down_overlap_record_batch(down_jobs.size());
        current_down_overlap_tensor_profile_record(
            rt.name, 1, down_jobs.size(), local_cache_hits, 0, local_missing_pack, 0);
        static std::atomic<int> first_current_down_overlap{0};
        if (first_current_down_overlap.fetch_add(1) == 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] current down overlap active: tensor=%s jobs=%zu bytes=%.2f MiB\n",
                rt.name, down_jobs.size(), rt.expert_bytes / (1024.0 * 1024.0));
        }

        current_down_overlap_thread = std::thread([&, down_cache, down_jobs = std::move(down_jobs), expert_bytes = rt.expert_bytes]() {
            const auto worker_start = std::chrono::steady_clock::now();
            bool copied = expert_pack_iouring_copy_jobs(down_jobs, expert_bytes, bc.prefetch_stream,
                    bc.stage_ring, "current_down_overlap");
            if (!copied) {
                copied = true;
                for (const stage_copy_job &job : down_jobs) {
                    batch_copy_trace copy_trace;
                    if (!batch_cache_copy_h2d(bc.stage_ring, job.dst, job.host_data, expert_bytes,
                            bc.prefetch_stream, job.pack_entry, &copy_trace,
                            "current_down_overlap", job.tensor, job.expert_idx)) {
                        copied = false;
                        break;
                    }
                }
            }

            if (copied) {
                for (const stage_copy_job &job : down_jobs) {
                    if (!down_cache->slot_ready[job.slot] &&
                            cudaEventCreateWithFlags(&down_cache->slot_ready[job.slot], cudaEventDisableTiming) != cudaSuccess) {
                        copied = false;
                        ++g_current_down_overlap.mark_failed;
                        break;
                    }
                    if (cudaEventRecord(down_cache->slot_ready[job.slot], bc.prefetch_stream) != cudaSuccess) {
                        copied = false;
                        ++g_current_down_overlap.mark_failed;
                        break;
                    }
                    down_cache->slot_pending[job.slot] = true;
                }
            }

            if (!copied) {
                cudaStreamSynchronize(bc.prefetch_stream);
                clear_stage_jobs_for_cache(down_cache, down_jobs);
                ++g_current_down_overlap.failed_batches;
                current_down_overlap_ok = false;
            } else {
                g_current_down_overlap.completed_jobs.fetch_add(down_jobs.size());
                current_down_overlap_tensor_profile_record(
                    down_jobs.empty() ? "" : down_jobs[0].tensor, 0, 0, 0, 0, 0, down_jobs.size());
            }

            const auto worker_end = std::chrono::steady_clock::now();
            g_current_down_overlap.worker_us.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(worker_end - worker_start).count());
        });
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

    auto submit_planned_host_prefetch = [&](const std::vector<stage_copy_job> &jobs) {
        for (const stage_copy_job &job : jobs) {
            if (job.pack_entry && job.tensor[0] && job.expert_idx >= 0) {
                host_prefetch_submit_planned(job.tensor, job.expert_idx, src0_bytes);
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

    if (exact_prompt_q8k) {
        if (ne00 % QK_K != 0) return false;
        const int nblocks = (int)(n_active * (ne00 / QK_K));
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, st>>>((const float *)bc.d_src1_f32, (block_q8_K *)bc.d_src1_q8k, nblocks);
        if (cudaGetLastError() != cudaSuccess) return false;
    } else if (fused_mmq_up_gate) {
        const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
        quantize_mmq_q8_1_cuda((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            src0_type, ne00, ne00, (int64_t)n_active * ne00, (int64_t)n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    if (profile) cudaEventRecord(bc.ev_quant, st);

    float * fused_d = use_handoff ? (float *)bc.d_handoff : (float *)bc.d_dst;
    const char *skip_nonresident_env = std::getenv("GGML_MOE_SKIP_NONRESIDENT");
    const bool skip_nonresident = skip_nonresident_env && skip_nonresident_env[0] && skip_nonresident_env[0] != '0';
    if (skip_nonresident) {
        if (cudaMemsetAsync(fused_d, 0, dst_bytes, st) != cudaSuccess) return false;
    }
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

    if (mixed_types) {
        static std::atomic<int> first_mixed_up_gate{0};
        if (first_mixed_up_gate.fetch_add(1) == 0) {
            std::fprintf(stderr,
                "[moe_stream] mixed-type up/gate compact path active: up_type=%d gate_type=%d up_bytes=%zu gate_bytes=%zu rows=%d\n",
                (int)src0_type, (int)gate_type, up_expert_bytes, gate_expert_bytes, n_active);
        }
        int up_slots[MOE_STREAM_MAX_ACTIVE] = {};
        if (!stage_tensor_slots_only_sized(
                up_key_name, src0_up_data, up_nb02, up_expert_bytes, st,
                bc.d_x_ids_up, bc.h_x_ids_up, up_slots, nullptr, 0)) {
            return decline("mixed_stage_up");
        }
        if (!stage_tensor_slots_only_sized(
                gate_key_name, src0_gate_data, gate_nb02, gate_expert_bytes, st,
                bc.d_x_ids_gate, bc.h_x_ids_gate, nullptr, up_slots, n_active)) {
            return decline("mixed_stage_gate");
        }
        if (cudaMemsetAsync(bc.d_up, 0, (size_t)n_active * (size_t)ne01 * sizeof(float), st) != cudaSuccess) {
            return decline("mixed_memset_up");
        }
        if (cudaMemsetAsync(bc.d_gate, 0, (size_t)n_active * (size_t)ne01 * sizeof(float), st) != cudaSuccess) {
            return decline("mixed_memset_gate");
        }
        if (!launch_moe_mmvq_compact_batch(
                src0_type,
                (const char *)cache->pool, bc.h_x_ids_up, cache->slot_sz,
                ne00, ne01, (const float *)bc.d_src1_f32, ne00, nullptr,
                bc.d_src1_q8_up, (float *)bc.d_up, n_active, st)) {
            return decline("mixed_launch_up");
        }
        if (profile) cudaEventRecord(bc.ev_up, st);
        if (!launch_moe_mmvq_compact_batch(
                gate_type,
                (const char *)cache->pool, bc.h_x_ids_gate, cache->slot_sz,
                ne00, ne01, (const float *)bc.d_src1_f32, ne00, nullptr,
                bc.d_src1_q8_gate, (float *)bc.d_gate, n_active, st)) {
            return decline("mixed_launch_gate");
        }
        if (profile) cudaEventRecord(bc.ev_gate, st);
        start_current_down_overlap();
        auto mixed_overlap_fail = [&](const char *reason) -> bool {
            (void)join_current_down_overlap();
            return decline(reason);
        };
        for (int j = 0; j < n_active; ++j) {
            bc.h_ids_dst[j] = flat_dst_ids[j];
        }
        if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) {
            return mixed_overlap_fail("mixed_copy_dst_ids");
        }
        if (cudaMemsetAsync(fused_d, 0, dst_bytes, st) != cudaSuccess) {
            return mixed_overlap_fail("mixed_memset_fused");
        }
        dim3 block(256);
        dim3 grid((unsigned int)((ne01 + block.x - 1) / block.x), (unsigned int)n_active);
        moe_stream_up_gate_fuse_kernel<<<grid, block, 0, st>>>(
            (const float *)bc.d_up, (const float *)bc.d_gate, fused_d,
            bc.d_ids_dst, n_active, ne01, unary_op, limit, true);
        if (cudaGetLastError() != cudaSuccess) {
            return mixed_overlap_fail("mixed_launch_fuse");
        }
        if (profile) cudaEventRecord(bc.ev_kernel, st);
        if (cudaMemcpyAsync(bc.h_dst, fused_d, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            return mixed_overlap_fail("mixed_copy_d2h");
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) {
            return mixed_overlap_fail("mixed_sync");
        }
        (void)join_current_down_overlap();
        const float *tmp = (const float *)bc.h_dst;
        for (int j = 0; j < n_active; ++j) {
            float *dst_row = (float *)((char *)dst + (size_t)dst_ids[j] * dst_nb1 + (size_t)token_ids[j] * dst_nb2);
            std::memcpy(dst_row, tmp + (size_t)flat_dst_ids[j] * ne01, (size_t)ne01 * sizeof(float));
        }
        return true;
    }

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

    if (fused_mmq_up_gate) {
        int up_slots[MOE_STREAM_MAX_ACTIVE] = {};
        if (!stage_tensor_slots_only(
                up_key_name, src0_up_data, st,
                bc.d_x_ids_up, bc.h_x_ids_up, up_slots, nullptr, 0)) {
            return false;
        }
        if (!stage_tensor_slots_only(
                gate_key_name, src0_gate_data, st,
                bc.d_x_ids_gate, bc.h_x_ids_gate, nullptr, up_slots, n_active)) {
            return false;
        }

        for (int j = 0; j < n_active; ++j) {
            bc.h_ids_dst[j] = j;
        }
        if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
        if (cudaMemsetAsync(bc.d_up, 0, (size_t)n_active * (size_t)ne01 * sizeof(float), st) != cudaSuccess) return false;
        if (cudaMemsetAsync(bc.d_gate, 0, (size_t)n_active * (size_t)ne01 * sizeof(float), st) != cudaSuccess) return false;
        if (!launch_moe_mmq_slot_batch(
                src0_type, (const char *)cache->pool, (const int *)bc.d_src1_q8,
                bc.d_ids_dst, bc.d_bounds, bc.d_x_ids_up, (float *)bc.d_up,
                ne00, ne01, nb01, cache->slot_sz, n_active, n_active, st)) {
            return false;
        }
        if (profile) cudaEventRecord(bc.ev_up, st);
        if (!launch_moe_mmq_slot_batch(
                src0_type, (const char *)cache->pool, (const int *)bc.d_src1_q8,
                bc.d_ids_dst, bc.d_bounds, bc.d_x_ids_gate, (float *)bc.d_gate,
                ne00, ne01, nb01, cache->slot_sz, n_active, n_active, st)) {
            return false;
        }
        if (profile) cudaEventRecord(bc.ev_gate, st);

        static std::atomic<int> mmq_compare_count{0};
        const int compare_call = mmq_compare ? mmq_compare_count.fetch_add(1) : INT_MAX;
        if (mmq_compare && compare_call < 4) {
            void *compare_buf = use_handoff ? bc.d_dst : bc.d_handoff;
            if (!use_handoff && !ensure_dev(bc.d_handoff, bc.d_handoff_sz, dst_bytes)) {
                return false;
            }
            compare_buf = use_handoff ? bc.d_dst : bc.d_handoff;
            const size_t compact_bytes = (size_t)n_active * (size_t)ne01 * sizeof(float);
            if (cudaMemsetAsync(compare_buf, 0, compact_bytes, st) != cudaSuccess) return false;
            if (!launch_moe_mmvq_compact_batch(
                    src0_type,
                    (const char *)cache->pool, bc.h_x_ids_up, cache->slot_sz,
                    ne00, ne01, (const float *)bc.d_src1_f32, ne00, nullptr,
                    bc.d_src1_q8_up, (float *)compare_buf, n_active, st)) {
                return false;
            }
            if (!moe_stream_compare_device_rows(
                    "up", (const float *)bc.d_up, (const float *)compare_buf,
                    bc.h_dst, bc.h_dst_sz, n_active, ne01, st)) {
                return false;
            }
            if (cudaMemsetAsync(compare_buf, 0, compact_bytes, st) != cudaSuccess) return false;
            if (!launch_moe_mmvq_compact_batch(
                    src0_type,
                    (const char *)cache->pool, bc.h_x_ids_gate, cache->slot_sz,
                    ne00, ne01, (const float *)bc.d_src1_f32, ne00, nullptr,
                    bc.d_src1_q8_gate, (float *)compare_buf, n_active, st)) {
                return false;
            }
            if (!moe_stream_compare_device_rows(
                    "gate", (const float *)bc.d_gate, (const float *)compare_buf,
                    bc.h_dst, bc.h_dst_sz, n_active, ne01, st)) {
                return false;
            }
        }

        for (int j = 0; j < n_active; ++j) {
            bc.h_ids_dst[j] = flat_dst_ids[j];
        }
        if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
        if (cudaMemsetAsync(fused_d, 0, dst_bytes, st) != cudaSuccess) return false;
    } else if (parallel_up_gate) {
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
            submit_planned_host_prefetch(up_jobs);
            submit_planned_host_prefetch(gate_jobs);

            const char *stage_split_env = std::getenv("GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT");
            if (!stage_split_env || !stage_split_env[0]) {
                stage_split_env = std::getenv("GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE");
            }
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
        if (serial_stage_batch) {
            std::vector<stage_copy_job> up_jobs;
            std::vector<stage_copy_job> gate_jobs;
            if (!plan_tensor(up_key_name, src0_up_data, bc.h_x_ids, nullptr, nullptr, 0, up_jobs)) {
                clear_stage_jobs(up_jobs);
                return false;
            }
            up_stage_jobs_count = (int)up_jobs.size();
            if (!copy_stage_jobs(up_jobs, st, bc.stage_ring)) {
                clear_stage_jobs(up_jobs);
                return false;
            }
            if (!launch_tensor(bc.d_up, st, bc.d_x_ids, bc.d_src1_q8, bc.h_x_ids)) {
                return false;
            }
            if (profile) cudaEventRecord(bc.ev_up, st);

            if (!plan_tensor(gate_key_name, src0_gate_data, bc.h_x_ids, nullptr, nullptr, 0, gate_jobs)) {
                clear_stage_jobs(gate_jobs);
                return false;
            }
            gate_stage_jobs_count = (int)gate_jobs.size();
            if (!copy_stage_jobs(gate_jobs, st, bc.stage_ring)) {
                clear_stage_jobs(gate_jobs);
                return false;
            }
            if (!launch_tensor(bc.d_gate, st, bc.d_x_ids, bc.d_src1_q8, bc.h_x_ids)) {
                return false;
            }
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
            up_gate_profile_add(
                g_uprof,
                (uint64_t)n_active,
                (uint64_t)up_stage_jobs_count,
                (uint64_t)gate_stage_jobs_count,
                stage_ms,
                quant_ms,
                up_ms,
                gate_ms,
                up_wait_ms,
                gate_wait_ms,
                up_compute_ms,
                gate_compute_ms,
                fuse_ms,
                kernel_ms,
                0.0,
                0.0,
                0.0);
            up_gate_type_profile_add(
                prompt_mode,
                src0_type,
                gate_type,
                (uint64_t)n_active,
                (uint64_t)up_stage_jobs_count,
                (uint64_t)gate_stage_jobs_count,
                stage_ms,
                quant_ms,
                up_ms,
                gate_ms,
                up_wait_ms,
                gate_wait_ms,
                up_compute_ms,
                gate_compute_ms,
                fuse_ms,
                kernel_ms,
                0.0,
                0.0,
                0.0);
            up_gate_layer_profile_add(
                prompt_mode,
                up_key_name,
                gate_key_name,
                src0_type,
                gate_type,
                (uint64_t)n_active,
                (uint64_t)up_stage_jobs_count,
                (uint64_t)gate_stage_jobs_count,
                stage_ms,
                quant_ms,
                up_ms,
                gate_ms,
                up_wait_ms,
                gate_wait_ms,
                up_compute_ms,
                gate_compute_ms,
                fuse_ms,
                kernel_ms,
                0.0,
                0.0,
                0.0);
            up_gate_profile_csv_record(
                prompt_mode,
                up_key_name,
                gate_key_name,
                src0_type,
                gate_type,
                n_active,
                n_active - up_stage_jobs_count,
                up_stage_jobs_count,
                n_active - gate_stage_jobs_count,
                gate_stage_jobs_count,
                up_stage_jobs_count,
                gate_stage_jobs_count,
                stage_ms,
                quant_ms,
                up_ms,
                gate_ms,
                up_wait_ms,
                gate_wait_ms,
                up_compute_ms,
                gate_compute_ms,
                fuse_ms,
                kernel_ms,
                0.0,
                0.0,
                0.0,
                use_handoff,
                parallel_up_gate,
                parallel_stage);
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
        const double wall_ms = std::chrono::duration<double, std::milli>(scatter_end - wall_start).count();
        up_gate_profile_add(
            g_uprof,
            (uint64_t)n_active,
            (uint64_t)up_stage_jobs_count,
            (uint64_t)gate_stage_jobs_count,
            stage_ms,
            quant_ms,
            up_ms,
            gate_ms,
            up_wait_ms,
            gate_wait_ms,
            up_compute_ms,
            gate_compute_ms,
            fuse_ms,
            kernel_ms,
            d2h_ms,
            scatter_ms,
            wall_ms);
        up_gate_type_profile_add(
            prompt_mode,
            src0_type,
            gate_type,
            (uint64_t)n_active,
            (uint64_t)up_stage_jobs_count,
            (uint64_t)gate_stage_jobs_count,
            stage_ms,
            quant_ms,
            up_ms,
            gate_ms,
            up_wait_ms,
            gate_wait_ms,
            up_compute_ms,
            gate_compute_ms,
            fuse_ms,
            kernel_ms,
            d2h_ms,
            scatter_ms,
            wall_ms);
        up_gate_layer_profile_add(
            prompt_mode,
            up_key_name,
            gate_key_name,
            src0_type,
            gate_type,
            (uint64_t)n_active,
            (uint64_t)up_stage_jobs_count,
            (uint64_t)gate_stage_jobs_count,
            stage_ms,
            quant_ms,
            up_ms,
            gate_ms,
            up_wait_ms,
            gate_wait_ms,
            up_compute_ms,
            gate_compute_ms,
            fuse_ms,
            kernel_ms,
            d2h_ms,
            scatter_ms,
            wall_ms);
        up_gate_profile_csv_record(
            prompt_mode,
            up_key_name,
            gate_key_name,
            src0_type,
            gate_type,
            n_active,
            n_active - up_stage_jobs_count,
            up_stage_jobs_count,
            n_active - gate_stage_jobs_count,
            gate_stage_jobs_count,
            up_stage_jobs_count,
            gate_stage_jobs_count,
            stage_ms,
            quant_ms,
            up_ms,
            gate_ms,
            up_wait_ms,
            gate_wait_ms,
            up_compute_ms,
            gate_compute_ms,
            fuse_ms,
            kernel_ms,
            d2h_ms,
            scatter_ms,
            wall_ms,
            use_handoff,
            parallel_up_gate,
            parallel_stage);
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
    ggml_type src0_type = (ggml_type)src0_type_int;
    const bool decline_debug = []() {
        const char *env = std::getenv("GGML_MOE_STREAM_DECLINE_DEBUG");
        return env && env[0] && env[0] != '0';
    }();
    int active_experts[128];
    int32_t dst_ids[128];
    int32_t token_ids[128];
    int n_active = 0;
    int max_dst_id = -1;
    auto decline = [&](const char *reason) -> bool {
        if (decline_debug) {
            std::fprintf(stderr,
                "[moe_stream_batch] down batch declined: reason=%s tensor=%s rows_stride=%ld type=%d active=%d ne01=%ld ne00=%ld\n",
                reason, src0_name ? src0_name : "", (long)rows_stride, src0_type_int,
                n_active, (long)ne01, (long)ne00);
        }
        return false;
    };
    if (!init_batch_once()) return decline("init_batch_once");
    if (!src0_name || !std::strstr(src0_name, "ffn_down_exps")) return decline("not_down_tensor");
    const char *down_q40_env = std::getenv("GGML_MOE_STREAM_DOWN_Q4_0");
    const char *down_q40_layer_range = std::getenv("GGML_MOE_STREAM_DOWN_Q4_0_LAYER_RANGE");
    const bool down_q40_requested =
        down_q40_env && down_q40_env[0] && down_q40_env[0] != '0' &&
        src0_type == GGML_TYPE_Q4_0 &&
        moe_tensor_layer_in_simple_range(src0_name, down_q40_layer_range);
    if (!moe_stream_type_supported(src0_type) && !down_q40_requested) return decline("unsupported_type");
    if (!src1_f32) return decline("missing_src1");
    ggml_cuda_moe_stream_register_tensor(src0_type_int, src0_name, src0_data, n_as, nb02, (size_t)ne01 * nb01);

    for (int64_t e = 0; e < n_as; ++e) {
        if (matrix_row_counts[e] != 1) {
            if (matrix_row_counts[e] > 1) return decline("multirow_not_supported");
            continue;
        }
        if (n_active >= 128) return decline("too_many_active_routes");
        const ggml_moe_row_mapping * r = matrix_rows + e*rows_stride;
        active_experts[n_active] = (int)e;
        dst_ids[n_active] = r[0].i1;
        token_ids[n_active] = r[0].i2;
        if (dst_ids[n_active] > max_dst_id) max_dst_id = dst_ids[n_active];
        ++n_active;
    }
    if (n_active <= 0 || max_dst_id < 0) return decline("no_active_routes");

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
    const auto wall_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
    const size_t src1_f32_bytes = (size_t)n_active * ne00 * sizeof(float);
    const size_t src1_q8_bytes = (size_t)n_active * moe_stream_q8_1_row_bytes(ne00);
    const char *down_q8k_env = std::getenv("GGML_MOE_STREAM_DOWN_Q8K");
    const char *down_q8k_types_env = std::getenv("GGML_MOE_STREAM_DOWN_Q8K_TYPES");
    const char *down_q8k_layer_range = std::getenv("GGML_MOE_STREAM_DOWN_Q8K_LAYER_RANGE");
    const bool down_q8k_type_allowed =
        !down_q8k_types_env || !down_q8k_types_env[0] ||
        std::strcmp(down_q8k_types_env, "all") == 0 ||
        (std::strcmp(down_q8k_types_env, "q3") == 0 && src0_type == GGML_TYPE_Q3_K) ||
        (std::strcmp(down_q8k_types_env, "iq4") == 0 && src0_type == GGML_TYPE_IQ4_XS);
    const bool down_q8k_candidate =
        down_q8k_env && down_q8k_env[0] && down_q8k_env[0] != '0' &&
        (src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ4_XS) &&
        down_q8k_type_allowed &&
        (!src0_name || std::strstr(src0_name, ".ffn_down_exps.") || std::strstr(src0_name, "ffn_down_exps")) &&
        moe_tensor_layer_in_simple_range(src0_name, down_q8k_layer_range);
    const int64_t dst_cols = max_dst_id + 1;
    const bool use_handoff =
        gpu_handoff_enabled() &&
        g_handoff.host_ptr == src1_f32 &&
        g_handoff.d_data &&
        g_handoff.ne01 == ne00 &&
        g_handoff.dst_cols >= dst_cols;
    const bool down_q8k_requested = down_q8k_candidate && !use_handoff;
    const size_t src1_q8k_bytes = down_q8k_requested ? (size_t)n_active * (ne00 / QK_K) * sizeof(block_q8_K) : 0;
    const size_t dst_bytes = (size_t)dst_cols * ne01 * sizeof(float);
    const size_t ids_bytes = (size_t)n_active * sizeof(int32_t);
    const size_t bounds_bytes = (size_t)(n_active + 1) * sizeof(int32_t);

    bool ok = ensure_dev(bc.d_src0, bc.d_src0_sz, src0_all_bytes)
        && (use_handoff || ensure_dev(bc.d_src1_f32, bc.d_src1_f32_sz, src1_f32_bytes))
        && ensure_dev(bc.d_src1_q8, bc.d_src1_q8_sz, src1_q8_bytes)
        && (!down_q8k_requested || ensure_dev(bc.d_src1_q8k, bc.d_src1_q8k_sz, src1_q8k_bytes))
        && ensure_dev(bc.d_dst, bc.d_dst_sz, dst_bytes)
        && ensure_dev((void *&)bc.d_ids_src1, bc.d_ids_src1_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_ids_dst, bc.d_ids_dst_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids, bc.d_x_ids_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_bounds, bc.d_bounds_sz, bounds_bytes)
        && (use_handoff || ensure_host_pinned(bc.h_src1, bc.h_src1_sz, src1_f32_bytes))
        && ensure_host_pinned(bc.h_dst, bc.h_dst_sz, dst_bytes);
    if (!ok) return decline("ensure_buffers");
    static std::atomic<int> first_down_q8k{0};
    if (down_q8k_requested && first_down_q8k.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream_batch] down Q3_K/IQ4_XS Q8_K-reference batch path active\n");
    }
    static std::atomic<int> first_down_q40{0};
    if (down_q40_requested && first_down_q40.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream_batch] down Q4_0 isolated cache MMVQ path active: range=%s\n",
                     down_q40_layer_range ? down_q40_layer_range : "");
    }

    batch_vram_cache *cache = down_q40_requested ?
        batch_cache_get_for_id(src0_bytes, BATCH_VRAM_CACHE_Q40_DOWN) :
        batch_cache_get(src0_bytes);
    if (!cache) return decline("cache_get");
    if (!down_q40_requested) {
        preload_profile_for_tensor(src0_name, src0_data, n_as, nb02, src0_bytes, st);
    }

    if (profile) cudaEventRecord(bc.ev_start, st);

    struct down_stage_copy_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
        int expert_idx = -1;
        char tensor[128] = {};
    };

    auto clear_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs) {
        for (const down_stage_copy_job &job : jobs) {
            batch_cache_clear_slot(cache, job.slot);
        }
    };

    auto copy_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs, cudaStream_t run_stream, pinned_stage_ring &ring) -> bool {
        if (cudaSetDevice(0) != cudaSuccess) return false;
        if (expert_pack_iouring_copy_jobs(jobs, src0_bytes, run_stream, ring, "runtime_load")) {
            return cudaGetLastError() == cudaSuccess;
        }
        for (const down_stage_copy_job &job : jobs) {
            batch_copy_trace copy_trace;
            const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry, &copy_trace,
                        "runtime_load", job.tensor, job.expert_idx)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr, &copy_trace,
                            "runtime_load", job.tensor, job.expert_idx)) {
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
    int down_profile_cache_hits = 0;
    int down_profile_cache_misses = 0;

    for (int j = 0; j < n_active; ++j) {
        const char *expert_host = (const char *)src0_data + (size_t)active_experts[j] * nb02;
        const uintptr_t cache_key = batch_key_hash(src0_name, active_experts[j]);
        batch_route_profile_hit(src0_name, active_experts[j], src0_bytes);
        int cache_slot = batch_cache_lookup_slot(cache, cache_key);
        if (cache_slot < 0) {
            ++down_profile_cache_misses;
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
            ++down_profile_cache_hits;
            batch_ttft_trace_record("cache_hit", src0_name, active_experts[j], src0_bytes, true, false, false, 0.0);
        }
        if (cache_slot < 0) return decline("cache_insert");
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
            return decline("parallel_stage_copy");
        }
        if (cudaEventRecord(bc.ev_up_done, bc.up_stream) != cudaSuccess) return decline("record_up_done");
        if (cudaEventRecord(bc.ev_gate_done, bc.gate_stream) != cudaSuccess) return decline("record_gate_done");
        if (cudaStreamWaitEvent(st, bc.ev_up_done, 0) != cudaSuccess) return decline("wait_up_done");
        if (cudaStreamWaitEvent(st, bc.ev_gate_done, 0) != cudaSuccess) return decline("wait_gate_done");
    }

    if (!use_handoff && cudaMemcpyAsync(bc.d_src1_f32, bc.h_src1, src1_f32_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return decline("copy_src1_h2d");
    if (cudaMemcpyAsync(bc.d_ids_src1, bc.h_ids_src1, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return decline("copy_ids_src1_h2d");
    if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return decline("copy_ids_dst_h2d");
    if (cudaMemcpyAsync(bc.d_x_ids, bc.h_x_ids, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return decline("copy_x_ids_h2d");
    if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, bounds_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return decline("copy_bounds_h2d");
    if (cudaMemsetAsync(bc.d_dst, 0, dst_bytes, st) != cudaSuccess) return decline("memset_dst");
    if (profile) cudaEventRecord(bc.ev_stage, st);

    if (use_handoff) {
        static std::atomic<int> first_handoff_consume{0};
        if (first_handoff_consume.fetch_add(1) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] GPU handoff consumed: ne00=%ld dst_cols=%ld\n",
                         (long)ne00, (long)dst_cols);
        }
    }
    if (profile) cudaEventRecord(bc.ev_quant, st);

    const float *d_src1_run = use_handoff ? g_handoff.d_data : (const float *)bc.d_src1_f32;
    const int64_t src1_run_stride = use_handoff ? g_handoff.ne01 : ne00;
    const int32_t *src1_rows = use_handoff ? bc.h_ids_dst : nullptr;
    if (down_q8k_requested) {
        if (ne00 % QK_K != 0) return decline("down_q8k_bad_ne00");
        const int nblocks = (int)(n_active * (ne00 / QK_K));
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, st>>>((const float *)bc.d_src1_f32, (block_q8_K *)bc.d_src1_q8k, nblocks);
        if (cudaGetLastError() != cudaSuccess) return decline("down_q8k_quantize");
        if (!launch_moe_iq3_xxs_q8k_batch(
                src0_type, (const char *)cache->pool, (const block_q8_K *)bc.d_src1_q8k,
                bc.d_ids_dst, bc.d_x_ids, (float *)bc.d_dst,
                ne00, ne01, nb01, cache->slot_sz, n_active, dst_cols, false, st)) {
            return decline("launch_down_q8k_batch");
        }
    } else if (!launch_moe_mmvq_compact_batch(
                src0_type,
                (const char *)cache->pool, bc.h_x_ids, cache->slot_sz,
                ne00, ne01, d_src1_run, src1_run_stride, src1_rows,
                bc.d_src1_q8, (float *)bc.d_dst, n_active, st)) {
            return decline("launch_moe_mmvq_compact_batch");
    }
    if (use_handoff) {
        g_handoff.host_ptr = nullptr;
        g_handoff.d_data = nullptr;
    }
    if (profile) cudaEventRecord(bc.ev_kernel, st);
    if (cudaMemcpyAsync(bc.h_dst, bc.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return decline("copy_dst_d2h");
    if (profile) cudaEventRecord(bc.ev_d2h, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return decline("sync_stream");

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
        std::memcpy(dst_row, tmp + (size_t)j * ne01, (size_t)ne01 * sizeof(float));
    }
    if (profile) {
        const auto scatter_end = std::chrono::steady_clock::now();
        const double scatter_ms = std::chrono::duration<double, std::milli>(scatter_end - scatter_start).count();
        const double wall_ms = std::chrono::duration<double, std::milli>(scatter_end - wall_start).count();
        ++g_bprof.calls;
        g_bprof.active_experts += (uint64_t)n_active;
        g_bprof.stage_ms += stage_ms;
        g_bprof.quant_ms += quant_ms;
        g_bprof.kernel_ms += kernel_ms;
        g_bprof.d2h_ms += d2h_ms;
        g_bprof.scatter_ms += scatter_ms;
        g_bprof.wall_ms += wall_ms;
        if (down_batch_profile_enabled()) {
            down_batch_profile_record(
                    src0_name,
                    src0_type,
                    n_active,
                    down_profile_cache_hits,
                    down_profile_cache_misses,
                    (int)(down_jobs_a.size() + down_jobs_b.size()),
                    stage_ms,
                    quant_ms,
                    kernel_ms,
                    d2h_ms,
                    scatter_ms,
                    wall_ms);
        }
    }
    return true;
}

#endif // GGML_CUDA_MOE_STREAM_BATCH
