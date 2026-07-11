// Default-off smoke test for running CUDA MMVQ on GGMLMOEPACKv2 entries.
//
// This validates that selected lower-byte payloads can be consumed by the
// exported MoE MMVQ kernel path. It does not load a model and does not change
// inference behavior.

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
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
    size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = split_tsv_line(line);
        const int need = std::max(std::max(c_tensor, c_expert), std::max(c_type, std::max(c_nbytes, std::max(c_ne00, std::max(c_ne01, c_nb01))))) + 1;
        if ((int)row.size() < need) {
            std::fprintf(stderr, "manifest short row line=%zu\n", line_no);
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
            std::fprintf(stderr, "manifest invalid row line=%zu\n", line_no);
            return false;
        }
        entries.push_back(std::move(e));
    }
    if (entries.empty()) {
        std::fprintf(stderr, "manifest no data rows: %s\n", path);
        return false;
    }
    return true;
}

static size_t q8_1_scratch_bytes(int64_t ne00) {
    const int64_t padded = ((ne00 + 31) / 32) * 32;
    // block_q8_1 is currently smaller than 64 bytes; this deliberately
    // over-allocates to avoid depending on private struct layout.
    return (size_t)(padded / 32) * 64;
}

static bool check_cuda(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool run_entry(lookup_fn lookup, read_fn read, mmvq_fn mmvq, const manifest_entry & expected, cudaStream_t stream) {
    int packed_type = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
    size_t nbytes = 0;
    if (!lookup(expected.tensor.c_str(), expected.expert_idx, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
        std::fprintf(stderr, "lookup failed tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
        return false;
    }
    if (packed_type != expected.packed_type || ne00 != expected.ne00 ||
            ne01 != expected.ne01 || nb01 != expected.nb01 || nbytes != expected.nbytes) {
        std::fprintf(stderr,
                "metadata mismatch tensor=%s expert=%d got type=%d ne00=%ld ne01=%ld nb01=%zu nbytes=%zu\n",
                expected.tensor.c_str(), expected.expert_idx,
                packed_type, (long)ne00, (long)ne01, nb01, nbytes);
        return false;
    }

    std::vector<uint8_t> host_src0(nbytes);
    size_t nread = 0;
    if (!read(expected.tensor.c_str(), expected.expert_idx, host_src0.data(), host_src0.size(), &nread) ||
            nread != nbytes) {
        std::fprintf(stderr, "read failed tensor=%s expert=%d nread=%zu nbytes=%zu\n",
                expected.tensor.c_str(), expected.expert_idx, nread, nbytes);
        return false;
    }

    std::vector<float> host_src1((size_t)ne00);
    for (int64_t i = 0; i < ne00; ++i) {
        host_src1[(size_t)i] = ((int)(i % 17) - 8) * 0.03125f;
    }
    std::vector<float> host_dst((size_t)ne01, 0.0f);

    void * d_src0 = nullptr;
    float * d_src1 = nullptr;
    void * d_src1_q8 = nullptr;
    float * d_dst = nullptr;
    const size_t src1_bytes = (size_t)ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)ne01 * sizeof(float);
    const size_t q8_bytes = q8_1_scratch_bytes(ne00);
    bool ok = true;
    ok = ok && check_cuda(cudaMalloc(&d_src0, nbytes), "cudaMalloc d_src0");
    ok = ok && check_cuda(cudaMalloc((void **)&d_src1, src1_bytes), "cudaMalloc d_src1");
    ok = ok && check_cuda(cudaMalloc(&d_src1_q8, q8_bytes), "cudaMalloc d_src1_q8");
    ok = ok && check_cuda(cudaMalloc((void **)&d_dst, dst_bytes), "cudaMalloc d_dst");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src0, host_src0.data(), nbytes, cudaMemcpyHostToDevice, stream), "copy src0 H2D");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src1, host_src1.data(), src1_bytes, cudaMemcpyHostToDevice, stream), "copy src1 H2D");
    ok = ok && check_cuda(cudaMemsetAsync(d_dst, 0, dst_bytes, stream), "memset dst");
    if (ok) {
        ok = mmvq(packed_type, d_src0, ne01, ne00, nb01, d_src1, d_src1_q8, d_dst, stream);
        if (!ok) {
            std::fprintf(stderr, "mmvq returned false tensor=%s expert=%d type=%d\n",
                    expected.tensor.c_str(), expected.expert_idx, packed_type);
        }
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "stream sync after mmvq");
    ok = ok && check_cuda(cudaMemcpy(host_dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "copy dst D2H");

    if (d_src0) cudaFree(d_src0);
    if (d_src1) cudaFree(d_src1);
    if (d_src1_q8) cudaFree(d_src1_q8);
    if (d_dst) cudaFree(d_dst);
    if (!ok) {
        return false;
    }

    double sum_abs = 0.0;
    double max_abs = 0.0;
    for (float v : host_dst) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr, "non-finite output tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
            return false;
        }
        const double av = std::fabs((double)v);
        sum_abs += av;
        if (av > max_abs) {
            max_abs = av;
        }
    }
    if (sum_abs == 0.0) {
        std::fprintf(stderr, "all-zero output tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
        return false;
    }
    std::printf("entry_mmvq_ok tensor=%s expert=%d type=%d ne00=%ld ne01=%ld nbytes=%zu sum_abs=%.6e max_abs=%.6e\n",
            expected.tensor.c_str(), expected.expert_idx, packed_type,
            (long)ne00, (long)ne01, nbytes, sum_abs, max_abs);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s LIBGGML_CUDA_SO V2_EXPERT_PACK MANIFEST_TSV\n", argv[0]);
        return 2;
    }
    const char * lib_path = argv[1];
    const char * pack_path = argv[2];
    const char * manifest_path = argv[3];

    if (setenv("GGML_MOE_EXPERT_PACK_V2", pack_path, 1) != 0) {
        std::perror("setenv GGML_MOE_EXPERT_PACK_V2");
        return 1;
    }

    std::vector<manifest_entry> entries;
    if (!load_manifest(manifest_path, entries)) {
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

    size_t ok_count = 0;
    for (const manifest_entry & entry : entries) {
        if (!run_entry(lookup, read, mmvq, entry, stream)) {
            cudaStreamDestroy(stream);
            dlclose(handle);
            return 1;
        }
        ++ok_count;
    }

    cudaStreamDestroy(stream);
    dlclose(handle);
    std::printf("kimi_moepack_v2_mmvq_smoke pass entries=%zu pack=%s manifest=%s\n",
            ok_count, pack_path, manifest_path);
    return 0;
}
