// Default-off smoke test for partial-covered GGMLMOEPACKv2 split dispatch.
//
// This validates the control-plane shape of a future runtime path:
// some active rows are computed from v2 lower-byte experts, uncovered rows keep
// a fallback result, and both are scattered back into one compact destination.
// It does not load a model and does not change inference behavior.

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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

struct route {
    bool covered = false;
    int expert_idx = -1;
    int dst_id = -1;
    manifest_entry entry;
    std::vector<float> values;
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
        std::vector<float> & out) {
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
        std::fprintf(stderr, "metadata mismatch tensor=%s expert=%d\n", expected.tensor.c_str(), expected.expert_idx);
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
        host_src1[(size_t)i] = ((int)(i % 23) - 11) * 0.025f;
    }

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
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "stream sync");

    out.assign((size_t)ne01, 0.0f);
    ok = ok && check_cuda(cudaMemcpy(out.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "copy dst D2H");

    if (d_src0) cudaFree(d_src0);
    if (d_src1) cudaFree(d_src1);
    if (d_src1_q8) cudaFree(d_src1_q8);
    if (d_dst) cudaFree(d_dst);
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
    for (int i = 0; i < n_rows; ++i) {
        route & r = routes[(size_t)i];
        float * dst_row = final_dst.data() + (size_t)r.dst_id * (size_t)base.ne01;
        if (r.covered) {
            if (!run_v2_entry(lookup, read, mmvq, r.entry, stream, r.values)) {
                cudaStreamDestroy(stream);
                dlclose(handle);
                return 1;
            }
            std::memcpy(dst_row, r.values.data(), (size_t)base.ne01 * sizeof(float));
        } else {
            r.values.resize((size_t)base.ne01);
            for (int64_t col = 0; col < base.ne01; ++col) {
                r.values[(size_t)col] = fallback_value(i, col);
            }
            std::memcpy(dst_row, r.values.data(), (size_t)base.ne01 * sizeof(float));
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
        "kimi_moepack_v2_partial_split_smoke pass tensor=%s routes=%d covered=%d fallback=%d ne00=%ld ne01=%ld pack=%s manifest=%s\n",
        base.tensor.c_str(), n_rows, covered_count, n_rows - covered_count,
        (long)base.ne00, (long)base.ne01, pack_path, manifest_path);
    for (int i = 0; i < n_rows; ++i) {
        const route & r = routes[(size_t)i];
        double sum_abs = 0.0;
        for (float v : r.values) {
            sum_abs += std::fabs((double)v);
        }
        std::printf("route row=%d dst=%d expert=%d covered=%d sum_abs=%.6e\n",
                i, r.dst_id, r.expert_idx, r.covered ? 1 : 0, sum_abs);
    }
    return 0;
}
