// Default-off smoke test for partial-covered GGMLMOEPACKv2 split dispatch.
//
// This validates the control-plane shape of a future runtime path:
// some active rows are computed from v2 lower-byte experts, uncovered rows keep
// a fallback result, and both are scattered back into one compact destination.
// It does not load a model and does not change inference behavior.

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using lookup_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        int * packed_type,
        int64_t * ne00,
        int64_t * ne01,
        size_t * nb01,
        size_t * nbytes);

using read_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        void * dst,
        size_t dst_capacity,
        size_t * nread);

using mmvq_fn = bool (*)(
        int src0_type_int,
        const void * d_src0,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float * d_src1_f32,
        void * d_src1_q8,
        float * d_dst,
        cudaStream_t stream);

struct manifest_entry {
    std::string tensor;
    int expert_idx = -1;
    int packed_type = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
    size_t nbytes = 0;
};

struct route_timing {
    bool covered = false;
    int expert_idx = -1;
    int dst_id = -1;
    int packed_type = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t payload_bytes = 0;
    size_t src1_bytes = 0;
    size_t dst_bytes = 0;
    size_t q8_bytes = 0;
    double lookup_us = 0.0;
    double host_src0_alloc_us = 0.0;
    double read_us = 0.0;
    double host_src1_us = 0.0;
    double cuda_alloc_us = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double sync_us = 0.0;
    double d2h_us = 0.0;
    double cuda_free_us = 0.0;
    double merge_us = 0.0;
    double fallback_fill_us = 0.0;
    double total_us = 0.0;
};

struct route {
    bool covered = false;
    int expert_idx = -1;
    int dst_id = -1;
    manifest_entry entry;
    std::vector<float> values;
    route_timing timing;
};

struct device_slot {
    void * d_src0 = nullptr;
    float * d_src1 = nullptr;
    void * d_src1_q8 = nullptr;
    float * d_dst = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
};

struct reuse_summary {
    int iterations = 0;
    int warm_iterations = 0;
    route_timing one_time;
    route_timing all_iters;
    route_timing warm_iters;
};

using steady_clock = std::chrono::steady_clock;

static double elapsed_us(
        const steady_clock::time_point & start,
        const steady_clock::time_point & end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static std::vector<std::string> split_tsv_line(const std::string & line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, '\t')) {
        out.push_back(field);
    }
    return out;
}

static bool load_manifest(const char * path, std::vector<manifest_entry> & entries) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "manifest open failed: %s\n", path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        std::fprintf(stderr, "manifest empty: %s\n", path);
        return false;
    }
    const std::vector<std::string> header = split_tsv_line(line);
    std::unordered_map<std::string, size_t> cols;
    for (size_t i = 0; i < header.size(); ++i) {
        cols[header[i]] = i;
    }
    auto col = [&](const char * name) -> int {
        auto it = cols.find(name);
        return it == cols.end() ? -1 : (int)it->second;
    };
    const int c_tensor = col("tensor");
    const int c_expert = col("expert_idx");
    const int c_type = col("packed_type");
    const int c_nbytes = col("packed_nbytes");
    const int c_ne00 = col("packed_ne00");
    const int c_ne01 = col("packed_ne01");
    const int c_nb01 = col("packed_nb01");
    if (c_tensor < 0 || c_expert < 0 || c_type < 0 || c_nbytes < 0 ||
            c_ne00 < 0 || c_ne01 < 0 || c_nb01 < 0) {
        std::fprintf(stderr, "manifest required columns missing: %s\n", path);
        return false;
    }

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = split_tsv_line(line);
        const int need = std::max(std::max(c_tensor, c_expert), std::max(c_type, std::max(c_nbytes, std::max(c_ne00, std::max(c_ne01, c_nb01))))) + 1;
        if ((int)row.size() < need) {
            return false;
        }
        manifest_entry e;
        e.tensor = row[(size_t)c_tensor];
        e.expert_idx = std::stoi(row[(size_t)c_expert]);
        e.packed_type = std::stoi(row[(size_t)c_type]);
        e.nbytes = (size_t)std::stoull(row[(size_t)c_nbytes]);
        e.ne00 = std::stoll(row[(size_t)c_ne00]);
        e.ne01 = std::stoll(row[(size_t)c_ne01]);
        e.nb01 = (size_t)std::stoull(row[(size_t)c_nb01]);
        if (e.tensor.empty() || e.expert_idx < 0 || e.packed_type <= 0 ||
                e.nbytes == 0 || e.ne00 <= 0 || e.ne01 <= 0 || e.nb01 == 0) {
            return false;
        }
        entries.push_back(std::move(e));
    }
    return !entries.empty();
}

static size_t q8_1_scratch_bytes(int64_t ne00) {
    const int64_t padded = ((ne00 + 31) / 32) * 32;
    return (size_t)(padded / 32) * 64;
}

static bool check_cuda(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool same_shape(const manifest_entry & a, const manifest_entry & b) {
    return a.tensor == b.tensor &&
        a.packed_type == b.packed_type &&
        a.ne00 == b.ne00 &&
        a.ne01 == b.ne01 &&
        a.nb01 == b.nb01 &&
        a.nbytes == b.nbytes;
}

static float fallback_value(int route_id, int64_t col) {
    return -1000.0f - (float)route_id * 0.25f - (float)(col % 17) * 0.001f;
}

static bool run_v2_entry(
        lookup_fn lookup,
        read_fn read,
        mmvq_fn mmvq,
        const manifest_entry & expected,
        cudaStream_t stream,
        std::vector<float> & out,
        route_timing & timing) {
    const steady_clock::time_point total_start = steady_clock::now();
    timing.covered = true;
    timing.expert_idx = expected.expert_idx;
    timing.packed_type = expected.packed_type;
    int packed_type = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
    size_t nbytes = 0;
    steady_clock::time_point t0 = steady_clock::now();
    if (!lookup(expected.tensor.c_str(), expected.expert_idx, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
        std::fprintf(stderr, "lookup failed tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
        return false;
    }
    timing.lookup_us = elapsed_us(t0, steady_clock::now());
    if (packed_type != expected.packed_type || ne00 != expected.ne00 ||
            ne01 != expected.ne01 || nb01 != expected.nb01 || nbytes != expected.nbytes) {
        std::fprintf(stderr, "metadata mismatch tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
        return false;
    }
    timing.ne00 = ne00;
    timing.ne01 = ne01;
    timing.payload_bytes = nbytes;

    t0 = steady_clock::now();
    std::vector<uint8_t> host_src0(nbytes);
    timing.host_src0_alloc_us = elapsed_us(t0, steady_clock::now());
    size_t nread = 0;
    t0 = steady_clock::now();
    if (!read(expected.tensor.c_str(), expected.expert_idx, host_src0.data(), host_src0.size(), &nread) ||
            nread != nbytes) {
        std::fprintf(stderr, "read failed tensor=%s expert=%d nread=%zu nbytes=%zu\n",
                expected.tensor.c_str(), expected.expert_idx, nread, nbytes);
        return false;
    }
    timing.read_us = elapsed_us(t0, steady_clock::now());

    t0 = steady_clock::now();
    std::vector<float> host_src1((size_t)ne00);
    for (int64_t i = 0; i < ne00; ++i) {
        host_src1[(size_t)i] = ((int)(i % 23) - 11) * 0.025f;
    }
    timing.host_src1_us = elapsed_us(t0, steady_clock::now());

    void * d_src0 = nullptr;
    float * d_src1 = nullptr;
    void * d_src1_q8 = nullptr;
    float * d_dst = nullptr;
    const size_t src1_bytes = (size_t)ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)ne01 * sizeof(float);
    const size_t q8_bytes = q8_1_scratch_bytes(ne00);
    timing.src1_bytes = src1_bytes;
    timing.dst_bytes = dst_bytes;
    timing.q8_bytes = q8_bytes;

    bool ok = true;
    t0 = steady_clock::now();
    ok = ok && check_cuda(cudaMalloc(&d_src0, nbytes), "cudaMalloc d_src0");
    ok = ok && check_cuda(cudaMalloc((void **)&d_src1, src1_bytes), "cudaMalloc d_src1");
    ok = ok && check_cuda(cudaMalloc(&d_src1_q8, q8_bytes), "cudaMalloc d_src1_q8");
    ok = ok && check_cuda(cudaMalloc((void **)&d_dst, dst_bytes), "cudaMalloc d_dst");
    timing.cuda_alloc_us = elapsed_us(t0, steady_clock::now());

    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    ok = ok && check_cuda(cudaEventCreate(&h2d_start), "cudaEventCreate h2d_start");
    ok = ok && check_cuda(cudaEventCreate(&h2d_stop), "cudaEventCreate h2d_stop");
    ok = ok && check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate kernel_start");
    ok = ok && check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate kernel_stop");
    ok = ok && check_cuda(cudaEventRecord(h2d_start, stream), "record h2d_start");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src0, host_src0.data(), nbytes, cudaMemcpyHostToDevice, stream), "copy src0 H2D");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src1, host_src1.data(), src1_bytes, cudaMemcpyHostToDevice, stream), "copy src1 H2D");
    ok = ok && check_cuda(cudaMemsetAsync(d_dst, 0, dst_bytes, stream), "memset dst");
    ok = ok && check_cuda(cudaEventRecord(h2d_stop, stream), "record h2d_stop");
    if (ok) {
        ok = ok && check_cuda(cudaEventRecord(kernel_start, stream), "record kernel_start");
        ok = mmvq(packed_type, d_src0, ne01, ne00, nb01, d_src1, d_src1_q8, d_dst, stream);
        ok = ok && check_cuda(cudaEventRecord(kernel_stop, stream), "record kernel_stop");
    }
    t0 = steady_clock::now();
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "stream sync");
    timing.sync_us = elapsed_us(t0, steady_clock::now());
    if (ok) {
        float h2d_elapsed_ms = 0.0f;
        float kernel_elapsed_ms = 0.0f;
        ok = ok && check_cuda(cudaEventElapsedTime(&h2d_elapsed_ms, h2d_start, h2d_stop), "elapsed h2d");
        ok = ok && check_cuda(cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop), "elapsed kernel");
        timing.h2d_ms = (double)h2d_elapsed_ms;
        timing.kernel_ms = (double)kernel_elapsed_ms;
    }

    out.assign((size_t)ne01, 0.0f);
    t0 = steady_clock::now();
    ok = ok && check_cuda(cudaMemcpy(out.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "copy dst D2H");
    timing.d2h_us = elapsed_us(t0, steady_clock::now());

    t0 = steady_clock::now();
    if (d_src0) cudaFree(d_src0);
    if (d_src1) cudaFree(d_src1);
    if (d_src1_q8) cudaFree(d_src1_q8);
    if (d_dst) cudaFree(d_dst);
    if (h2d_start) cudaEventDestroy(h2d_start);
    if (h2d_stop) cudaEventDestroy(h2d_stop);
    if (kernel_start) cudaEventDestroy(kernel_start);
    if (kernel_stop) cudaEventDestroy(kernel_stop);
    timing.cuda_free_us = elapsed_us(t0, steady_clock::now());
    timing.total_us = elapsed_us(total_start, steady_clock::now());
    if (!ok) {
        return false;
    }

    double sum_abs = 0.0;
    for (float v : out) {
        if (!std::isfinite(v)) {
            return false;
        }
        sum_abs += std::fabs((double)v);
    }
    return sum_abs > 0.0;
}

static void free_device_slot(device_slot & slot) {
    if (slot.d_src0) cudaFree(slot.d_src0);
    if (slot.d_src1) cudaFree(slot.d_src1);
    if (slot.d_src1_q8) cudaFree(slot.d_src1_q8);
    if (slot.d_dst) cudaFree(slot.d_dst);
    if (slot.h2d_start) cudaEventDestroy(slot.h2d_start);
    if (slot.h2d_stop) cudaEventDestroy(slot.h2d_stop);
    if (slot.kernel_start) cudaEventDestroy(slot.kernel_start);
    if (slot.kernel_stop) cudaEventDestroy(slot.kernel_stop);
    slot = device_slot{};
}

static bool run_v2_entries_batched(
        lookup_fn lookup,
        read_fn read,
        mmvq_fn mmvq,
        std::vector<route> & routes,
        const manifest_entry & base,
        cudaStream_t stream,
        std::vector<float> & final_dst,
        route_timing & batch_timing,
        bool pinned_host) {
    const steady_clock::time_point total_start = steady_clock::now();
    std::vector<int> covered_route_ids;
    for (int i = 0; i < (int)routes.size(); ++i) {
        if (routes[(size_t)i].covered) {
            covered_route_ids.push_back(i);
        }
    }
    if (covered_route_ids.empty()) {
        return false;
    }

    const size_t n_covered = covered_route_ids.size();
    const size_t src1_bytes = (size_t)base.ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)base.ne01 * sizeof(float);
    const size_t q8_bytes = q8_1_scratch_bytes(base.ne00);

    std::vector<std::vector<uint8_t>> host_src0_owned(n_covered);
    std::vector<void *> host_src0_pinned(n_covered, nullptr);
    std::vector<uint8_t *> host_src0_ptrs(n_covered, nullptr);
    std::vector<float> host_src1_owned;
    void * host_src1_pinned = nullptr;
    float * host_src1_ptr = nullptr;
    std::vector<device_slot> slots(n_covered);

    auto free_host_buffers = [&]() {
        for (void *& p : host_src0_pinned) {
            if (p) {
                cudaFreeHost(p);
                p = nullptr;
            }
        }
        if (host_src1_pinned) {
            cudaFreeHost(host_src1_pinned);
            host_src1_pinned = nullptr;
        }
    };

    bool ok = true;
    steady_clock::time_point t0;

    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        route_timing & timing = r.timing;
        timing = route_timing{};
        timing.covered = true;
        timing.expert_idx = r.expert_idx;
        timing.dst_id = r.dst_id;
        timing.packed_type = r.entry.packed_type;

        int packed_type = 0;
        int64_t ne00 = 0;
        int64_t ne01 = 0;
        size_t nb01 = 0;
        size_t nbytes = 0;
        t0 = steady_clock::now();
        if (!lookup(r.entry.tensor.c_str(), r.entry.expert_idx, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
            std::fprintf(stderr, "batch lookup failed tensor=%s expert=%d\n", r.entry.tensor.c_str(), r.entry.expert_idx);
            free_host_buffers();
            return false;
        }
        timing.lookup_us = elapsed_us(t0, steady_clock::now());
        if (packed_type != r.entry.packed_type || ne00 != r.entry.ne00 ||
                ne01 != r.entry.ne01 || nb01 != r.entry.nb01 || nbytes != r.entry.nbytes) {
            std::fprintf(stderr, "batch metadata mismatch tensor=%s expert=%d\n", r.entry.tensor.c_str(), r.entry.expert_idx);
            free_host_buffers();
            return false;
        }
        timing.ne00 = ne00;
        timing.ne01 = ne01;
        timing.payload_bytes = nbytes;
        timing.src1_bytes = src1_bytes;
        timing.dst_bytes = dst_bytes;
        timing.q8_bytes = q8_bytes;

        t0 = steady_clock::now();
        if (pinned_host) {
            void * p = nullptr;
            if (!check_cuda(cudaHostAlloc(&p, nbytes, cudaHostAllocDefault), "batch cudaHostAlloc src0")) {
                timing.host_src0_alloc_us = elapsed_us(t0, steady_clock::now());
                free_host_buffers();
                return false;
            }
            host_src0_pinned[j] = p;
            host_src0_ptrs[j] = (uint8_t *)p;
        } else {
            host_src0_owned[j].resize(nbytes);
            host_src0_ptrs[j] = host_src0_owned[j].data();
        }
        timing.host_src0_alloc_us = elapsed_us(t0, steady_clock::now());
        size_t nread = 0;
        t0 = steady_clock::now();
        if (!read(r.entry.tensor.c_str(), r.entry.expert_idx, host_src0_ptrs[j], nbytes, &nread) ||
                nread != nbytes) {
            std::fprintf(stderr, "batch read failed tensor=%s expert=%d nread=%zu nbytes=%zu\n",
                    r.entry.tensor.c_str(), r.entry.expert_idx, nread, nbytes);
            free_host_buffers();
            return false;
        }
        timing.read_us = elapsed_us(t0, steady_clock::now());
    }

    t0 = steady_clock::now();
    if (pinned_host) {
        if (!check_cuda(cudaHostAlloc(&host_src1_pinned, src1_bytes, cudaHostAllocDefault), "batch cudaHostAlloc src1")) {
            batch_timing.host_src1_us = elapsed_us(t0, steady_clock::now());
            free_host_buffers();
            return false;
        }
        host_src1_ptr = (float *)host_src1_pinned;
    } else {
        host_src1_owned.resize((size_t)base.ne00);
        host_src1_ptr = host_src1_owned.data();
    }
    for (int64_t i = 0; i < base.ne00; ++i) {
        host_src1_ptr[(size_t)i] = ((int)(i % 23) - 11) * 0.025f;
    }
    batch_timing.host_src1_us = elapsed_us(t0, steady_clock::now());

    t0 = steady_clock::now();
    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        device_slot & slot = slots[j];
        ok = ok && check_cuda(cudaMalloc(&slot.d_src0, r.entry.nbytes), "batch cudaMalloc d_src0");
        ok = ok && check_cuda(cudaMalloc((void **)&slot.d_src1, src1_bytes), "batch cudaMalloc d_src1");
        ok = ok && check_cuda(cudaMalloc(&slot.d_src1_q8, q8_bytes), "batch cudaMalloc d_src1_q8");
        ok = ok && check_cuda(cudaMalloc((void **)&slot.d_dst, dst_bytes), "batch cudaMalloc d_dst");
        ok = ok && check_cuda(cudaEventCreate(&slot.h2d_start), "batch cudaEventCreate h2d_start");
        ok = ok && check_cuda(cudaEventCreate(&slot.h2d_stop), "batch cudaEventCreate h2d_stop");
        ok = ok && check_cuda(cudaEventCreate(&slot.kernel_start), "batch cudaEventCreate kernel_start");
        ok = ok && check_cuda(cudaEventCreate(&slot.kernel_stop), "batch cudaEventCreate kernel_stop");
        if (!ok) {
            break;
        }
    }
    batch_timing.cuda_alloc_us = elapsed_us(t0, steady_clock::now());
    if (!ok) {
        for (device_slot & slot : slots) {
            free_device_slot(slot);
        }
        free_host_buffers();
        return false;
    }

    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        device_slot & slot = slots[j];
        ok = ok && check_cuda(cudaEventRecord(slot.h2d_start, stream), "batch record h2d_start");
        ok = ok && check_cuda(cudaMemcpyAsync(slot.d_src0, host_src0_ptrs[j], r.entry.nbytes, cudaMemcpyHostToDevice, stream), "batch copy src0 H2D");
        ok = ok && check_cuda(cudaMemcpyAsync(slot.d_src1, host_src1_ptr, src1_bytes, cudaMemcpyHostToDevice, stream), "batch copy src1 H2D");
        ok = ok && check_cuda(cudaMemsetAsync(slot.d_dst, 0, dst_bytes, stream), "batch memset dst");
        ok = ok && check_cuda(cudaEventRecord(slot.h2d_stop, stream), "batch record h2d_stop");
        ok = ok && check_cuda(cudaEventRecord(slot.kernel_start, stream), "batch record kernel_start");
        if (ok) {
            ok = mmvq(r.entry.packed_type, slot.d_src0, r.entry.ne01, r.entry.ne00, r.entry.nb01,
                    slot.d_src1, slot.d_src1_q8, slot.d_dst, stream);
        }
        ok = ok && check_cuda(cudaEventRecord(slot.kernel_stop, stream), "batch record kernel_stop");
        if (!ok) {
            break;
        }
    }

    t0 = steady_clock::now();
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "batch stream sync");
    batch_timing.sync_us = elapsed_us(t0, steady_clock::now());
    if (ok) {
        for (size_t j = 0; j < n_covered; ++j) {
            route & r = routes[(size_t)covered_route_ids[j]];
            device_slot & slot = slots[j];
            float h2d_elapsed_ms = 0.0f;
            float kernel_elapsed_ms = 0.0f;
            ok = ok && check_cuda(cudaEventElapsedTime(&h2d_elapsed_ms, slot.h2d_start, slot.h2d_stop), "batch elapsed h2d");
            ok = ok && check_cuda(cudaEventElapsedTime(&kernel_elapsed_ms, slot.kernel_start, slot.kernel_stop), "batch elapsed kernel");
            r.timing.h2d_ms = (double)h2d_elapsed_ms;
            r.timing.kernel_ms = (double)kernel_elapsed_ms;
            batch_timing.h2d_ms += r.timing.h2d_ms;
            batch_timing.kernel_ms += r.timing.kernel_ms;
        }
    }
    if (!ok) {
        for (device_slot & slot : slots) {
            free_device_slot(slot);
        }
        free_host_buffers();
        return false;
    }

    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        device_slot & slot = slots[j];
        r.values.assign((size_t)base.ne01, 0.0f);
        t0 = steady_clock::now();
        ok = ok && check_cuda(cudaMemcpy(r.values.data(), slot.d_dst, dst_bytes, cudaMemcpyDeviceToHost), "batch copy dst D2H");
        r.timing.d2h_us = elapsed_us(t0, steady_clock::now());
        if (!ok) {
            break;
        }
        double sum_abs = 0.0;
        for (float v : r.values) {
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
            sum_abs += std::fabs((double)v);
        }
        if (!ok || sum_abs <= 0.0) {
            std::fprintf(stderr, "batch covered output invalid route=%d expert=%d\n",
                    covered_route_ids[j], r.expert_idx);
            ok = false;
            break;
        }
        float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
        t0 = steady_clock::now();
        std::memcpy(dst_row, r.values.data(), dst_bytes);
        r.timing.merge_us = elapsed_us(t0, steady_clock::now());
    }

    for (int i = 0; ok && i < (int)routes.size(); ++i) {
        route & r = routes[(size_t)i];
        if (r.covered) {
            continue;
        }
        r.timing = route_timing{};
        r.timing.covered = false;
        r.timing.expert_idx = r.expert_idx;
        r.timing.dst_id = r.dst_id;
        t0 = steady_clock::now();
        r.values.resize((size_t)base.ne01);
        for (int64_t col = 0; col < base.ne01; ++col) {
            r.values[(size_t)col] = fallback_value(i, col);
        }
        r.timing.fallback_fill_us = elapsed_us(t0, steady_clock::now());
        float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
        t0 = steady_clock::now();
        std::memcpy(dst_row, r.values.data(), dst_bytes);
        r.timing.merge_us = elapsed_us(t0, steady_clock::now());
        r.timing.total_us = r.timing.fallback_fill_us + r.timing.merge_us;
    }

    t0 = steady_clock::now();
    for (device_slot & slot : slots) {
        free_device_slot(slot);
    }
    free_host_buffers();
    batch_timing.cuda_free_us = elapsed_us(t0, steady_clock::now());

    for (int id : covered_route_ids) {
        route_timing & t = routes[(size_t)id].timing;
        t.total_us = t.lookup_us + t.host_src0_alloc_us + t.read_us +
            t.h2d_ms * 1000.0 + t.kernel_ms * 1000.0 + t.d2h_us + t.merge_us;
        batch_timing.payload_bytes += t.payload_bytes;
        batch_timing.src1_bytes += t.src1_bytes;
        batch_timing.dst_bytes += t.dst_bytes;
        batch_timing.q8_bytes += t.q8_bytes;
        batch_timing.lookup_us += t.lookup_us;
        batch_timing.host_src0_alloc_us += t.host_src0_alloc_us;
        batch_timing.read_us += t.read_us;
        batch_timing.d2h_us += t.d2h_us;
        batch_timing.merge_us += t.merge_us;
    }
    for (const route & r : routes) {
        if (!r.covered) {
            batch_timing.fallback_fill_us += r.timing.fallback_fill_us;
            batch_timing.merge_us += r.timing.merge_us;
        }
    }
    batch_timing.total_us = elapsed_us(total_start, steady_clock::now());
    if (!ok) {
        return false;
    }
    return true;
}

static bool run_v2_entries_batched_reuse(
        lookup_fn lookup,
        read_fn read,
        mmvq_fn mmvq,
        std::vector<route> & routes,
        const manifest_entry & base,
        cudaStream_t stream,
        std::vector<float> & final_dst,
        int iterations,
        reuse_summary & summary,
        std::vector<route_timing> & iter_timings) {
    std::vector<int> covered_route_ids;
    for (int i = 0; i < (int)routes.size(); ++i) {
        if (routes[(size_t)i].covered) {
            covered_route_ids.push_back(i);
        }
    }
    if (covered_route_ids.empty() || iterations <= 0) {
        return false;
    }

    const size_t n_covered = covered_route_ids.size();
    const size_t src1_bytes = (size_t)base.ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)base.ne01 * sizeof(float);
    const size_t q8_bytes = q8_1_scratch_bytes(base.ne00);

    std::vector<void *> host_src0_pinned(n_covered, nullptr);
    std::vector<uint8_t *> host_src0_ptrs(n_covered, nullptr);
    void * host_src1_pinned = nullptr;
    float * host_src1_ptr = nullptr;
    std::vector<device_slot> slots(n_covered);

    auto cleanup = [&]() {
        for (device_slot & slot : slots) {
            free_device_slot(slot);
        }
        for (void *& p : host_src0_pinned) {
            if (p) {
                cudaFreeHost(p);
                p = nullptr;
            }
        }
        if (host_src1_pinned) {
            cudaFreeHost(host_src1_pinned);
            host_src1_pinned = nullptr;
        }
    };

    bool ok = true;
    steady_clock::time_point t0;

    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        route_timing & timing = r.timing;
        timing = route_timing{};
        timing.covered = true;
        timing.expert_idx = r.expert_idx;
        timing.dst_id = r.dst_id;
        timing.packed_type = r.entry.packed_type;

        int packed_type = 0;
        int64_t ne00 = 0;
        int64_t ne01 = 0;
        size_t nb01 = 0;
        size_t nbytes = 0;
        t0 = steady_clock::now();
        if (!lookup(r.entry.tensor.c_str(), r.entry.expert_idx, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
            std::fprintf(stderr, "reuse lookup failed tensor=%s expert=%d\n", r.entry.tensor.c_str(), r.entry.expert_idx);
            cleanup();
            return false;
        }
        timing.lookup_us = elapsed_us(t0, steady_clock::now());
        summary.one_time.lookup_us += timing.lookup_us;
        if (packed_type != r.entry.packed_type || ne00 != r.entry.ne00 ||
                ne01 != r.entry.ne01 || nb01 != r.entry.nb01 || nbytes != r.entry.nbytes) {
            std::fprintf(stderr, "reuse metadata mismatch tensor=%s expert=%d\n", r.entry.tensor.c_str(), r.entry.expert_idx);
            cleanup();
            return false;
        }
        timing.ne00 = ne00;
        timing.ne01 = ne01;
        timing.payload_bytes = nbytes;
        timing.src1_bytes = src1_bytes;
        timing.dst_bytes = dst_bytes;
        timing.q8_bytes = q8_bytes;

        t0 = steady_clock::now();
        void * p = nullptr;
        if (!check_cuda(cudaHostAlloc(&p, nbytes, cudaHostAllocDefault), "reuse cudaHostAlloc src0")) {
            timing.host_src0_alloc_us = elapsed_us(t0, steady_clock::now());
            cleanup();
            return false;
        }
        timing.host_src0_alloc_us = elapsed_us(t0, steady_clock::now());
        summary.one_time.host_src0_alloc_us += timing.host_src0_alloc_us;
        host_src0_pinned[j] = p;
        host_src0_ptrs[j] = (uint8_t *)p;
    }

    t0 = steady_clock::now();
    if (!check_cuda(cudaHostAlloc(&host_src1_pinned, src1_bytes, cudaHostAllocDefault), "reuse cudaHostAlloc src1")) {
        summary.one_time.host_src1_us = elapsed_us(t0, steady_clock::now());
        cleanup();
        return false;
    }
    host_src1_ptr = (float *)host_src1_pinned;
    for (int64_t i = 0; i < base.ne00; ++i) {
        host_src1_ptr[(size_t)i] = ((int)(i % 23) - 11) * 0.025f;
    }
    summary.one_time.host_src1_us = elapsed_us(t0, steady_clock::now());

    t0 = steady_clock::now();
    for (size_t j = 0; j < n_covered; ++j) {
        route & r = routes[(size_t)covered_route_ids[j]];
        device_slot & slot = slots[j];
        ok = ok && check_cuda(cudaMalloc(&slot.d_src0, r.entry.nbytes), "reuse cudaMalloc d_src0");
        ok = ok && check_cuda(cudaMalloc((void **)&slot.d_src1, src1_bytes), "reuse cudaMalloc d_src1");
        ok = ok && check_cuda(cudaMalloc(&slot.d_src1_q8, q8_bytes), "reuse cudaMalloc d_src1_q8");
        ok = ok && check_cuda(cudaMalloc((void **)&slot.d_dst, dst_bytes), "reuse cudaMalloc d_dst");
        ok = ok && check_cuda(cudaEventCreate(&slot.h2d_start), "reuse cudaEventCreate h2d_start");
        ok = ok && check_cuda(cudaEventCreate(&slot.h2d_stop), "reuse cudaEventCreate h2d_stop");
        ok = ok && check_cuda(cudaEventCreate(&slot.kernel_start), "reuse cudaEventCreate kernel_start");
        ok = ok && check_cuda(cudaEventCreate(&slot.kernel_stop), "reuse cudaEventCreate kernel_stop");
        if (!ok) {
            break;
        }
    }
    summary.one_time.cuda_alloc_us = elapsed_us(t0, steady_clock::now());
    if (!ok) {
        cleanup();
        return false;
    }

    iter_timings.clear();
    iter_timings.reserve((size_t)iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        route_timing it;
        it.covered = true;
        const steady_clock::time_point iter_start = steady_clock::now();

        for (size_t j = 0; j < n_covered; ++j) {
            route & r = routes[(size_t)covered_route_ids[j]];
            size_t nread = 0;
            t0 = steady_clock::now();
            if (!read(r.entry.tensor.c_str(), r.entry.expert_idx, host_src0_ptrs[j], r.entry.nbytes, &nread) ||
                    nread != r.entry.nbytes) {
                std::fprintf(stderr, "reuse read failed iter=%d tensor=%s expert=%d nread=%zu nbytes=%zu\n",
                        iter, r.entry.tensor.c_str(), r.entry.expert_idx, nread, r.entry.nbytes);
                cleanup();
                return false;
            }
            const double read_us = elapsed_us(t0, steady_clock::now());
            routes[(size_t)covered_route_ids[j]].timing.read_us = read_us;
            it.read_us += read_us;
            it.payload_bytes += r.entry.nbytes;
        }

        for (size_t j = 0; j < n_covered; ++j) {
            route & r = routes[(size_t)covered_route_ids[j]];
            device_slot & slot = slots[j];
            ok = ok && check_cuda(cudaEventRecord(slot.h2d_start, stream), "reuse record h2d_start");
            ok = ok && check_cuda(cudaMemcpyAsync(slot.d_src0, host_src0_ptrs[j], r.entry.nbytes, cudaMemcpyHostToDevice, stream), "reuse copy src0 H2D");
            ok = ok && check_cuda(cudaMemcpyAsync(slot.d_src1, host_src1_ptr, src1_bytes, cudaMemcpyHostToDevice, stream), "reuse copy src1 H2D");
            ok = ok && check_cuda(cudaMemsetAsync(slot.d_dst, 0, dst_bytes, stream), "reuse memset dst");
            ok = ok && check_cuda(cudaEventRecord(slot.h2d_stop, stream), "reuse record h2d_stop");
            ok = ok && check_cuda(cudaEventRecord(slot.kernel_start, stream), "reuse record kernel_start");
            if (ok) {
                ok = mmvq(r.entry.packed_type, slot.d_src0, r.entry.ne01, r.entry.ne00, r.entry.nb01,
                        slot.d_src1, slot.d_src1_q8, slot.d_dst, stream);
            }
            ok = ok && check_cuda(cudaEventRecord(slot.kernel_stop, stream), "reuse record kernel_stop");
            if (!ok) {
                break;
            }
        }
        t0 = steady_clock::now();
        ok = ok && check_cuda(cudaStreamSynchronize(stream), "reuse stream sync");
        it.sync_us = elapsed_us(t0, steady_clock::now());
        if (!ok) {
            cleanup();
            return false;
        }

        for (size_t j = 0; j < n_covered; ++j) {
            route & r = routes[(size_t)covered_route_ids[j]];
            device_slot & slot = slots[j];
            float h2d_elapsed_ms = 0.0f;
            float kernel_elapsed_ms = 0.0f;
            ok = ok && check_cuda(cudaEventElapsedTime(&h2d_elapsed_ms, slot.h2d_start, slot.h2d_stop), "reuse elapsed h2d");
            ok = ok && check_cuda(cudaEventElapsedTime(&kernel_elapsed_ms, slot.kernel_start, slot.kernel_stop), "reuse elapsed kernel");
            r.timing.h2d_ms = (double)h2d_elapsed_ms;
            r.timing.kernel_ms = (double)kernel_elapsed_ms;
            it.h2d_ms += r.timing.h2d_ms;
            it.kernel_ms += r.timing.kernel_ms;
        }
        if (!ok) {
            cleanup();
            return false;
        }

        for (size_t j = 0; j < n_covered; ++j) {
            route & r = routes[(size_t)covered_route_ids[j]];
            device_slot & slot = slots[j];
            r.values.assign((size_t)base.ne01, 0.0f);
            t0 = steady_clock::now();
            ok = ok && check_cuda(cudaMemcpy(r.values.data(), slot.d_dst, dst_bytes, cudaMemcpyDeviceToHost), "reuse copy dst D2H");
            const double d2h_us = elapsed_us(t0, steady_clock::now());
            r.timing.d2h_us = d2h_us;
            it.d2h_us += d2h_us;
            if (!ok) {
                cleanup();
                return false;
            }
            double sum_abs = 0.0;
            for (float v : r.values) {
                if (!std::isfinite(v)) {
                    ok = false;
                    break;
                }
                sum_abs += std::fabs((double)v);
            }
            if (!ok || sum_abs <= 0.0) {
                std::fprintf(stderr, "reuse covered output invalid iter=%d route=%d expert=%d\n",
                        iter, covered_route_ids[j], r.expert_idx);
                cleanup();
                return false;
            }
            float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
            t0 = steady_clock::now();
            std::memcpy(dst_row, r.values.data(), dst_bytes);
            const double merge_us = elapsed_us(t0, steady_clock::now());
            r.timing.merge_us = merge_us;
            r.timing.total_us = r.timing.read_us + r.timing.h2d_ms * 1000.0 +
                r.timing.kernel_ms * 1000.0 + r.timing.d2h_us + r.timing.merge_us;
            it.merge_us += merge_us;
        }

        for (int i = 0; i < (int)routes.size(); ++i) {
            route & r = routes[(size_t)i];
            if (r.covered) {
                continue;
            }
            r.timing = route_timing{};
            r.timing.covered = false;
            r.timing.expert_idx = r.expert_idx;
            r.timing.dst_id = r.dst_id;
            t0 = steady_clock::now();
            r.values.resize((size_t)base.ne01);
            for (int64_t col = 0; col < base.ne01; ++col) {
                r.values[(size_t)col] = fallback_value(i, col);
            }
            r.timing.fallback_fill_us = elapsed_us(t0, steady_clock::now());
            float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
            t0 = steady_clock::now();
            std::memcpy(dst_row, r.values.data(), dst_bytes);
            r.timing.merge_us = elapsed_us(t0, steady_clock::now());
            r.timing.total_us = r.timing.fallback_fill_us + r.timing.merge_us;
            it.fallback_fill_us += r.timing.fallback_fill_us;
            it.merge_us += r.timing.merge_us;
        }

        it.total_us = elapsed_us(iter_start, steady_clock::now());
        iter_timings.push_back(it);
        summary.all_iters.read_us += it.read_us;
        summary.all_iters.h2d_ms += it.h2d_ms;
        summary.all_iters.kernel_ms += it.kernel_ms;
        summary.all_iters.sync_us += it.sync_us;
        summary.all_iters.d2h_us += it.d2h_us;
        summary.all_iters.merge_us += it.merge_us;
        summary.all_iters.fallback_fill_us += it.fallback_fill_us;
        summary.all_iters.total_us += it.total_us;
        summary.all_iters.payload_bytes += it.payload_bytes;
        if (iter > 0) {
            summary.warm_iters.read_us += it.read_us;
            summary.warm_iters.h2d_ms += it.h2d_ms;
            summary.warm_iters.kernel_ms += it.kernel_ms;
            summary.warm_iters.sync_us += it.sync_us;
            summary.warm_iters.d2h_us += it.d2h_us;
            summary.warm_iters.merge_us += it.merge_us;
            summary.warm_iters.fallback_fill_us += it.fallback_fill_us;
            summary.warm_iters.total_us += it.total_us;
            summary.warm_iters.payload_bytes += it.payload_bytes;
        }
    }

    t0 = steady_clock::now();
    cleanup();
    summary.one_time.cuda_free_us = elapsed_us(t0, steady_clock::now());
    summary.iterations = iterations;
    summary.warm_iterations = std::max(0, iterations - 1);
    return true;
}

static bool choose_group(
        const std::vector<manifest_entry> & entries,
        std::vector<manifest_entry> & chosen) {
    std::unordered_map<std::string, std::vector<manifest_entry>> groups;
    for (const manifest_entry & e : entries) {
        groups[e.tensor].push_back(e);
    }

    std::string best_name;
    size_t best_size = 0;
    for (auto & kv : groups) {
        if (kv.second.size() < 2) {
            continue;
        }
        const bool prefer_up = kv.first.find("ffn_up_exps") != std::string::npos;
        const size_t score = kv.second.size() + (prefer_up ? 1000 : 0);
        if (score > best_size) {
            best_size = score;
            best_name = kv.first;
        }
    }
    if (best_name.empty()) {
        return false;
    }
    chosen = groups[best_name];
    std::sort(chosen.begin(), chosen.end(), [](const manifest_entry & a, const manifest_entry & b) {
        return a.expert_idx < b.expert_idx;
    });
    const manifest_entry base = chosen[0];
    chosen.erase(std::remove_if(chosen.begin(), chosen.end(), [&](const manifest_entry & e) {
        return !same_shape(base, e);
    }), chosen.end());
    return chosen.size() >= 2;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s LIBGGML_CUDA_SO V2_EXPERT_PACK MANIFEST_TSV [per-row|batch|batch-pinned|batch-pinned-reuse]\n", argv[0]);
        return 2;
    }
    const char * lib_path = argv[1];
    const char * pack_path = argv[2];
    const char * manifest_path = argv[3];
    const char * mode = argc == 5 ? argv[4] : "per-row";
    const bool batch_mode = std::strcmp(mode, "batch") == 0;
    const bool batch_pinned_mode = std::strcmp(mode, "batch-pinned") == 0;
    const bool batch_pinned_reuse_mode = std::strcmp(mode, "batch-pinned-reuse") == 0;
    const bool batch_like_mode = batch_mode || batch_pinned_mode || batch_pinned_reuse_mode;
    const bool per_row_mode = std::strcmp(mode, "per-row") == 0;
    if (!batch_like_mode && !per_row_mode) {
        std::fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }
    if (setenv("GGML_MOE_EXPERT_PACK_V2", pack_path, 1) != 0) {
        std::perror("setenv GGML_MOE_EXPERT_PACK_V2");
        return 1;
    }

    std::vector<manifest_entry> entries;
    if (!load_manifest(manifest_path, entries)) {
        return 1;
    }
    std::vector<manifest_entry> group;
    if (!choose_group(entries, group)) {
        std::fprintf(stderr, "no tensor group with at least two same-shape v2 entries\n");
        return 1;
    }

    void * handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    auto lookup = (lookup_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_lookup_debug");
    auto read = (read_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_read_debug");
    auto mmvq = (mmvq_fn)dlsym(handle, "ggml_cuda_moe_stream_mmvq_dev");
    if (!lookup || !read || !mmvq) {
        std::fprintf(stderr, "required symbols missing lookup=%p read=%p mmvq=%p\n",
                (void *)lookup, (void *)read, (void *)mmvq);
        dlclose(handle);
        return 1;
    }
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) {
        dlclose(handle);
        return 1;
    }
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
        dlclose(handle);
        return 1;
    }

    const int covered_count = std::min<int>(3, (int)group.size());
    std::unordered_set<int> covered_experts;
    for (int i = 0; i < covered_count; ++i) {
        covered_experts.insert(group[(size_t)i].expert_idx);
    }
    int fallback_a = 0;
    while (covered_experts.count(fallback_a)) ++fallback_a;
    int fallback_b = fallback_a + 1;
    while (covered_experts.count(fallback_b)) ++fallback_b;

    std::vector<route> routes;
    auto add_covered = [&](int src, int dst) {
        route r;
        r.covered = true;
        r.entry = group[(size_t)src];
        r.expert_idx = r.entry.expert_idx;
        r.dst_id = dst;
        routes.push_back(std::move(r));
    };
    auto add_fallback = [&](int expert, int dst) {
        route r;
        r.covered = false;
        r.expert_idx = expert;
        r.dst_id = dst;
        routes.push_back(std::move(r));
    };

    add_covered(0, 3);
    add_fallback(fallback_a, 0);
    add_covered(1, 4);
    add_fallback(fallback_b, 1);
    if (covered_count >= 3) {
        add_covered(2, 2);
    }

    const manifest_entry & base = group[0];
    const int n_rows = (int)routes.size();
    std::vector<float> final_dst((size_t)n_rows * (size_t)base.ne01, std::numeric_limits<float>::quiet_NaN());
    route_timing batch_timing;
    reuse_summary reuse_timing;
    std::vector<route_timing> reuse_iter_timings;
    if (batch_pinned_reuse_mode) {
        if (!run_v2_entries_batched_reuse(lookup, read, mmvq, routes, base, stream, final_dst, 5, reuse_timing, reuse_iter_timings)) {
            cudaStreamDestroy(stream);
            dlclose(handle);
            return 1;
        }
    } else if (batch_like_mode) {
        if (!run_v2_entries_batched(lookup, read, mmvq, routes, base, stream, final_dst, batch_timing, batch_pinned_mode)) {
            cudaStreamDestroy(stream);
            dlclose(handle);
            return 1;
        }
    } else {
        for (int i = 0; i < n_rows; ++i) {
            route & r = routes[(size_t)i];
            float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
            r.timing.covered = r.covered;
            r.timing.expert_idx = r.expert_idx;
            r.timing.dst_id = r.dst_id;
            if (r.covered) {
                if (!run_v2_entry(lookup, read, mmvq, r.entry, stream, r.values, r.timing)) {
                    cudaStreamDestroy(stream);
                    dlclose(handle);
                    return 1;
                }
                const steady_clock::time_point merge_start = steady_clock::now();
                std::memcpy(dst_row, r.values.data(), (size_t)base.ne01 * sizeof(float));
                r.timing.merge_us = elapsed_us(merge_start, steady_clock::now());
                r.timing.total_us += r.timing.merge_us;
            } else {
                const steady_clock::time_point fill_start = steady_clock::now();
                r.values.resize((size_t)base.ne01);
                for (int64_t col = 0; col < base.ne01; ++col) {
                    r.values[(size_t)col] = fallback_value(i, col);
                }
                r.timing.fallback_fill_us = elapsed_us(fill_start, steady_clock::now());
                const steady_clock::time_point merge_start = steady_clock::now();
                std::memcpy(dst_row, r.values.data(), (size_t)base.ne01 * sizeof(float));
                r.timing.merge_us = elapsed_us(merge_start, steady_clock::now());
                r.timing.total_us = r.timing.fallback_fill_us + r.timing.merge_us;
            }
        }
    }

    bool ok = true;
    for (int i = 0; i < n_rows; ++i) {
        const route & r = routes[(size_t)i];
        const float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
        const int64_t probes[] = {0, 1, base.ne01 / 2, base.ne01 - 1};
        for (int64_t col : probes) {
            const float got = dst_row[(size_t)col];
            const float want = r.values[(size_t)col];
            if (std::fabs(got - want) > 1e-6f) {
                std::fprintf(stderr,
                        "row placement mismatch route=%d covered=%d expert=%d dst=%d col=%ld got=%g want=%g\n",
                        i, r.covered ? 1 : 0, r.expert_idx, r.dst_id, (long)col, (double)got, (double)want);
                ok = false;
            }
        }
    }
    for (size_t i = 0; i < final_dst.size(); ++i) {
        if (!std::isfinite(final_dst[i])) {
            std::fprintf(stderr, "final dst has non-finite at flat index=%zu\n", i);
            ok = false;
            break;
        }
    }

    cudaStreamDestroy(stream);
    dlclose(handle);
    if (!ok) {
        return 1;
    }

    std::printf(
        "kimi_moepack_v2_partial_split_smoke pass mode=%s tensor=%s routes=%d covered=%d fallback=%d ne00=%ld ne01=%ld pack=%s manifest=%s\n",
        mode, base.tensor.c_str(), n_rows, covered_count, n_rows - covered_count,
        (long)base.ne00, (long)base.ne01, pack_path, manifest_path);
    route_timing total;
    int measured_covered = 0;
    int measured_fallback = 0;
    for (const route & r : routes) {
        const route_timing & t = r.timing;
        if (r.covered) {
            ++measured_covered;
            total.payload_bytes += t.payload_bytes;
            total.src1_bytes += t.src1_bytes;
            total.dst_bytes += t.dst_bytes;
            total.q8_bytes += t.q8_bytes;
            total.lookup_us += t.lookup_us;
            total.host_src0_alloc_us += t.host_src0_alloc_us;
            total.read_us += t.read_us;
            total.host_src1_us += t.host_src1_us;
            total.cuda_alloc_us += t.cuda_alloc_us;
            total.h2d_ms += t.h2d_ms;
            total.kernel_ms += t.kernel_ms;
            total.sync_us += t.sync_us;
            total.d2h_us += t.d2h_us;
            total.cuda_free_us += t.cuda_free_us;
            total.merge_us += t.merge_us;
            total.total_us += t.total_us;
        } else {
            ++measured_fallback;
            total.fallback_fill_us += t.fallback_fill_us;
            total.merge_us += t.merge_us;
            total.total_us += t.total_us;
        }
    }
    const double payload_mib = (double)total.payload_bytes / 1048576.0;
    std::printf(
        "timing_summary covered=%d fallback=%d payload_mib=%.3f lookup_us=%.3f host_src0_alloc_us=%.3f read_us=%.3f host_src1_us=%.3f cuda_alloc_us=%.3f h2d_ms=%.3f kernel_ms=%.3f sync_us=%.3f d2h_us=%.3f cuda_free_us=%.3f merge_us=%.3f fallback_fill_us=%.3f total_us=%.3f\n",
        measured_covered, measured_fallback, payload_mib, total.lookup_us,
        total.host_src0_alloc_us, total.read_us, total.host_src1_us,
        total.cuda_alloc_us, total.h2d_ms, total.kernel_ms, total.sync_us,
        total.d2h_us, total.cuda_free_us, total.merge_us,
        total.fallback_fill_us, total.total_us);
    if (measured_covered > 0) {
        std::printf(
            "timing_per_covered payload_mib=%.3f lookup_us=%.3f read_us=%.3f cuda_alloc_us=%.3f h2d_ms=%.3f kernel_ms=%.3f sync_us=%.3f d2h_us=%.3f cuda_free_us=%.3f merge_us=%.3f total_us=%.3f\n",
            payload_mib / (double)measured_covered,
            total.lookup_us / (double)measured_covered,
            total.read_us / (double)measured_covered,
            total.cuda_alloc_us / (double)measured_covered,
            total.h2d_ms / (double)measured_covered,
            total.kernel_ms / (double)measured_covered,
            total.sync_us / (double)measured_covered,
            total.d2h_us / (double)measured_covered,
            total.cuda_free_us / (double)measured_covered,
            total.merge_us / (double)n_rows,
            total.total_us / (double)measured_covered);
    }
    if (batch_mode || batch_pinned_mode) {
        std::printf(
            "timing_batch_summary covered=%d fallback=%d payload_mib=%.3f lookup_us=%.3f host_src0_alloc_us=%.3f read_us=%.3f host_src1_us=%.3f cuda_alloc_us=%.3f h2d_ms=%.3f kernel_ms=%.3f batch_sync_us=%.3f d2h_us=%.3f cuda_free_us=%.3f merge_us=%.3f fallback_fill_us=%.3f total_us=%.3f\n",
            measured_covered, measured_fallback, (double)batch_timing.payload_bytes / 1048576.0,
            batch_timing.lookup_us, batch_timing.host_src0_alloc_us, batch_timing.read_us,
            batch_timing.host_src1_us, batch_timing.cuda_alloc_us, batch_timing.h2d_ms,
            batch_timing.kernel_ms, batch_timing.sync_us, batch_timing.d2h_us,
            batch_timing.cuda_free_us, batch_timing.merge_us,
            batch_timing.fallback_fill_us, batch_timing.total_us);
    }
    if (batch_pinned_reuse_mode) {
        for (size_t i = 0; i < reuse_iter_timings.size(); ++i) {
            const route_timing & it = reuse_iter_timings[i];
            std::printf(
                "timing_reuse_iter iter=%zu payload_mib=%.3f read_us=%.3f h2d_ms=%.3f kernel_ms=%.3f sync_us=%.3f d2h_us=%.3f merge_us=%.3f fallback_fill_us=%.3f total_us=%.3f\n",
                i, (double)it.payload_bytes / 1048576.0, it.read_us, it.h2d_ms,
                it.kernel_ms, it.sync_us, it.d2h_us, it.merge_us,
                it.fallback_fill_us, it.total_us);
        }
        const double warm_div = reuse_timing.warm_iterations > 0 ? (double)reuse_timing.warm_iterations : 1.0;
        std::printf(
            "timing_reuse_summary iterations=%d warm_iterations=%d payload_mib_per_iter=%.3f one_time_lookup_us=%.3f one_time_host_src0_alloc_us=%.3f one_time_host_src1_us=%.3f one_time_cuda_alloc_us=%.3f one_time_cuda_free_us=%.3f all_total_us=%.3f warm_avg_total_us=%.3f warm_avg_read_us=%.3f warm_avg_h2d_ms=%.3f warm_avg_kernel_ms=%.3f warm_avg_sync_us=%.3f warm_avg_d2h_us=%.3f warm_avg_merge_us=%.3f warm_avg_fallback_fill_us=%.3f\n",
            reuse_timing.iterations, reuse_timing.warm_iterations,
            reuse_iter_timings.empty() ? 0.0 : (double)reuse_iter_timings[0].payload_bytes / 1048576.0,
            reuse_timing.one_time.lookup_us,
            reuse_timing.one_time.host_src0_alloc_us,
            reuse_timing.one_time.host_src1_us,
            reuse_timing.one_time.cuda_alloc_us,
            reuse_timing.one_time.cuda_free_us,
            reuse_timing.all_iters.total_us,
            reuse_timing.warm_iters.total_us / warm_div,
            reuse_timing.warm_iters.read_us / warm_div,
            reuse_timing.warm_iters.h2d_ms / warm_div,
            reuse_timing.warm_iters.kernel_ms / warm_div,
            reuse_timing.warm_iters.sync_us / warm_div,
            reuse_timing.warm_iters.d2h_us / warm_div,
            reuse_timing.warm_iters.merge_us / warm_div,
            reuse_timing.warm_iters.fallback_fill_us / warm_div);
    }
    for (int i = 0; i < n_rows; ++i) {
        const route & r = routes[(size_t)i];
        double sum_abs = 0.0;
        for (float v : r.values) {
            sum_abs += std::fabs((double)v);
        }
        std::printf("route row=%d dst=%d expert=%d covered=%d sum_abs=%.6e\n",
                i, r.dst_id, r.expert_idx, r.covered ? 1 : 0, sum_abs);
        const route_timing & t = r.timing;
        std::printf(
            "timing_route row=%d dst=%d expert=%d covered=%d payload_bytes=%zu lookup_us=%.3f host_src0_alloc_us=%.3f read_us=%.3f host_src1_us=%.3f cuda_alloc_us=%.3f h2d_ms=%.3f kernel_ms=%.3f sync_us=%.3f d2h_us=%.3f cuda_free_us=%.3f merge_us=%.3f fallback_fill_us=%.3f total_us=%.3f\n",
            i, r.dst_id, r.expert_idx, r.covered ? 1 : 0, t.payload_bytes,
            t.lookup_us, t.host_src0_alloc_us, t.read_us, t.host_src1_us,
            t.cuda_alloc_us, t.h2d_ms, t.kernel_ms, t.sync_us, t.d2h_us,
            t.cuda_free_us, t.merge_us, t.fallback_fill_us, t.total_us);
    }
    return 0;
}
