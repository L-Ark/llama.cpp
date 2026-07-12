// Default-off smoke test comparing current IQ3-pack expert MMVQ output with a
// GGMLMOEPACKv2 lower-byte overlay for the same tensor/expert.
//
// This does not load a model and does not change inference behavior. It is an
// activation-output quality gate for tiny lower-byte payloads before wiring any
// runtime override path.

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using v2_lookup_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        int * packed_type,
        int64_t * ne00,
        int64_t * ne01,
        size_t * nb01,
        size_t * nbytes);

using v2_read_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        void * dst,
        size_t dst_capacity,
        size_t * nread);

using current_read_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        void * dst,
        size_t dst_capacity,
        size_t * nread,
        int * source_is_gguf);

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
    std::string kind;
    std::string remote_type;
    int packed_type = 0;
    size_t packed_nbytes = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t packed_nb01 = 0;
    size_t current_nbytes = 0;
};

struct inventory_entry {
    std::string current_type_name;
    int current_type = -1;
};

struct metrics {
    double rel_l2_sum = 0.0;
    double rel_l2_max = 0.0;
    double cosine_sum = 0.0;
    double mean_abs_sum = 0.0;
    double max_abs = 0.0;
    double current_l2_sum = 0.0;
    int rows = 0;
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

static std::unordered_map<std::string, size_t> header_cols(const std::vector<std::string> & header) {
    std::unordered_map<std::string, size_t> cols;
    for (size_t i = 0; i < header.size(); ++i) {
        cols[header[i]] = i;
    }
    return cols;
}

static bool get_col(
        const std::vector<std::string> & row,
        const std::unordered_map<std::string, size_t> & cols,
        const char * name,
        std::string & out) {
    const auto it = cols.find(name);
    if (it == cols.end() || it->second >= row.size()) {
        return false;
    }
    out = row[it->second];
    return true;
}

static int type_name_to_int(const std::string & name) {
    if (name == "F32") return 0;
    if (name == "F16") return 1;
    if (name == "Q4_0") return 2;
    if (name == "Q4_1") return 3;
    if (name == "Q5_0") return 6;
    if (name == "Q5_1") return 7;
    if (name == "Q8_0") return 8;
    if (name == "Q8_1") return 9;
    if (name == "Q2_K") return 10;
    if (name == "Q3_K") return 11;
    if (name == "Q4_K") return 12;
    if (name == "Q5_K") return 13;
    if (name == "Q6_K") return 14;
    if (name == "Q8_K") return 15;
    if (name == "IQ2_XXS") return 16;
    if (name == "IQ2_XS") return 17;
    if (name == "IQ3_XXS") return 18;
    if (name == "IQ1_S") return 19;
    if (name == "IQ4_NL") return 20;
    if (name == "IQ3_S") return 21;
    if (name == "IQ2_S") return 22;
    if (name == "IQ4_XS") return 23;
    return -1;
}

static bool load_inventory(const char * path, std::unordered_map<std::string, inventory_entry> & out) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "inventory open failed: %s\n", path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        std::fprintf(stderr, "inventory empty: %s\n", path);
        return false;
    }
    const auto cols = header_cols(split_tsv_line(line));
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto row = split_tsv_line(line);
        std::string tensor;
        std::string type;
        if (!get_col(row, cols, "tensor", tensor) || !get_col(row, cols, "type", type)) {
            std::fprintf(stderr, "inventory required columns missing\n");
            return false;
        }
        inventory_entry entry;
        entry.current_type_name = type;
        entry.current_type = type_name_to_int(type);
        if (entry.current_type < 0) {
            std::fprintf(stderr, "unsupported current type in inventory: %s tensor=%s\n", type.c_str(), tensor.c_str());
            return false;
        }
        out[tensor] = entry;
    }
    return !out.empty();
}

static bool load_manifest(const char * path, std::vector<manifest_entry> & entries, int max_entries) {
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
    const auto cols = header_cols(split_tsv_line(line));
    auto require = [&](const char * name) -> bool {
        if (cols.find(name) == cols.end()) {
            std::fprintf(stderr, "manifest required column missing: %s\n", name);
            return false;
        }
        return true;
    };
    const char * required[] = {
        "tensor", "expert_idx", "kind", "remote_type", "packed_type",
        "packed_nbytes", "packed_ne00", "packed_ne01", "packed_nb01",
        "current_nbytes",
    };
    for (const char * name : required) {
        if (!require(name)) {
            return false;
        }
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto row = split_tsv_line(line);
        std::string v;
        manifest_entry e;
        get_col(row, cols, "tensor", e.tensor);
        get_col(row, cols, "kind", e.kind);
        get_col(row, cols, "remote_type", e.remote_type);
        get_col(row, cols, "expert_idx", v); e.expert_idx = std::stoi(v);
        get_col(row, cols, "packed_type", v); e.packed_type = std::stoi(v);
        get_col(row, cols, "packed_nbytes", v); e.packed_nbytes = (size_t)std::stoull(v);
        get_col(row, cols, "packed_ne00", v); e.ne00 = std::stoll(v);
        get_col(row, cols, "packed_ne01", v); e.ne01 = std::stoll(v);
        get_col(row, cols, "packed_nb01", v); e.packed_nb01 = (size_t)std::stoull(v);
        get_col(row, cols, "current_nbytes", v); e.current_nbytes = (size_t)std::stoull(v);
        if (e.tensor.empty() || e.expert_idx < 0 || e.packed_type <= 0 ||
                e.packed_nbytes == 0 || e.current_nbytes == 0 ||
                e.ne00 <= 0 || e.ne01 <= 0 || e.packed_nb01 == 0) {
            std::fprintf(stderr, "invalid manifest row: %s\n", line.c_str());
            return false;
        }
        entries.push_back(std::move(e));
        if (max_entries > 0 && (int)entries.size() >= max_entries) {
            break;
        }
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

static bool run_mmvq(
        mmvq_fn mmvq,
        int type,
        const std::vector<uint8_t> & src0,
        int64_t ne00,
        int64_t ne01,
        size_t nb01,
        const std::vector<float> & src1,
        std::vector<float> & dst,
        cudaStream_t stream) {
    void * d_src0 = nullptr;
    float * d_src1 = nullptr;
    void * d_src1_q8 = nullptr;
    float * d_dst = nullptr;
    const size_t src1_bytes = (size_t)ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)ne01 * sizeof(float);
    const size_t q8_bytes = q8_1_scratch_bytes(ne00);
    bool ok = true;
    ok = ok && check_cuda(cudaMalloc(&d_src0, src0.size()), "cudaMalloc d_src0");
    ok = ok && check_cuda(cudaMalloc((void **)&d_src1, src1_bytes), "cudaMalloc d_src1");
    ok = ok && check_cuda(cudaMalloc(&d_src1_q8, q8_bytes), "cudaMalloc d_src1_q8");
    ok = ok && check_cuda(cudaMalloc((void **)&d_dst, dst_bytes), "cudaMalloc d_dst");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src0, src0.data(), src0.size(), cudaMemcpyHostToDevice, stream), "copy src0 H2D");
    ok = ok && check_cuda(cudaMemcpyAsync(d_src1, src1.data(), src1_bytes, cudaMemcpyHostToDevice, stream), "copy src1 H2D");
    ok = ok && check_cuda(cudaMemsetAsync(d_dst, 0, dst_bytes, stream), "memset dst");
    if (ok) {
        ok = mmvq(type, d_src0, ne01, ne00, nb01, d_src1, d_src1_q8, d_dst, stream);
        if (!ok) {
            std::fprintf(stderr, "mmvq returned false type=%d ne00=%ld ne01=%ld nb01=%zu\n",
                    type, (long)ne00, (long)ne01, nb01);
        }
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "stream sync after mmvq");
    dst.assign((size_t)ne01, 0.0f);
    ok = ok && check_cuda(cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "copy dst D2H");
    if (d_src0) cudaFree(d_src0);
    if (d_src1) cudaFree(d_src1);
    if (d_src1_q8) cudaFree(d_src1_q8);
    if (d_dst) cudaFree(d_dst);
    return ok;
}

static void add_metrics(metrics & m, const std::vector<float> & current, const std::vector<float> & lower) {
    double diff2 = 0.0;
    double cur2 = 0.0;
    double low2 = 0.0;
    double dot = 0.0;
    double abs_sum = 0.0;
    double abs_max = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
        const double c = current[i];
        const double l = lower[i];
        if (!std::isfinite(c) || !std::isfinite(l)) {
            diff2 = INFINITY;
            break;
        }
        const double d = l - c;
        diff2 += d * d;
        cur2 += c * c;
        low2 += l * l;
        dot += c * l;
        const double ad = std::fabs(d);
        abs_sum += ad;
        abs_max = std::max(abs_max, ad);
    }
    const double cur_l2 = std::sqrt(std::max(cur2, 0.0));
    const double low_l2 = std::sqrt(std::max(low2, 0.0));
    const double rel = std::sqrt(diff2) / std::max(cur_l2, 1e-30);
    const double cosine = dot / std::max(cur_l2 * low_l2, 1e-30);
    m.rel_l2_sum += rel;
    m.rel_l2_max = std::max(m.rel_l2_max, rel);
    m.cosine_sum += cosine;
    m.mean_abs_sum += abs_sum / std::max<size_t>(current.size(), 1);
    m.max_abs = std::max(m.max_abs, abs_max);
    m.current_l2_sum += cur_l2;
    m.rows += 1;
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr,
                "usage: %s LIBGGML_CUDA_SO V2_EXPERT_PACK MANIFEST_TSV INVENTORY_TSV [MAX_ENTRIES=0] [ACTIVATIONS_PER_ENTRY=4]\n",
                argv[0]);
        return 2;
    }
    const char * lib_path = argv[1];
    const char * v2_pack_path = argv[2];
    const char * manifest_path = argv[3];
    const char * inventory_path = argv[4];
    const int max_entries = argc >= 6 ? std::atoi(argv[5]) : 0;
    const int activations_per_entry = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 4;

    if (setenv("GGML_MOE_EXPERT_PACK_V2", v2_pack_path, 1) != 0) {
        std::perror("setenv GGML_MOE_EXPERT_PACK_V2");
        return 1;
    }

    std::vector<manifest_entry> entries;
    std::unordered_map<std::string, inventory_entry> inventory;
    if (!load_manifest(manifest_path, entries, max_entries) ||
            !load_inventory(inventory_path, inventory)) {
        return 1;
    }

    void * handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    auto lookup_v2 = (v2_lookup_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_lookup_debug");
    auto read_v2 = (v2_read_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_read_debug");
    auto read_current = (current_read_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_read_to_host");
    auto mmvq = (mmvq_fn)dlsym(handle, "ggml_cuda_moe_stream_mmvq_dev");
    if (!lookup_v2 || !read_v2 || !read_current || !mmvq) {
        std::fprintf(stderr, "required symbols missing: lookup_v2=%p read_v2=%p read_current=%p mmvq=%p\n",
                (void *)lookup_v2, (void *)read_v2, (void *)read_current, (void *)mmvq);
        dlclose(handle);
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        dlclose(handle);
        return 1;
    }

    std::printf("tensor\texpert_idx\tkind\tcurrent_type\tremote_type\tcurrent_nbytes\tpacked_nbytes\tbyte_ratio\tactivations\tmean_rel_l2\tmax_rel_l2\tmean_cosine\tmean_abs_error\tmax_abs_error\tmean_current_l2\tcurrent_source_is_gguf\n");

    metrics total;
    bool ok_all = true;
    for (const manifest_entry & entry : entries) {
        const auto inv_it = inventory.find(entry.tensor);
        if (inv_it == inventory.end()) {
            std::fprintf(stderr, "tensor not found in inventory: %s\n", entry.tensor.c_str());
            ok_all = false;
            continue;
        }
        const inventory_entry & inv = inv_it->second;
        const size_t current_nb01 = entry.current_nbytes / (size_t)entry.ne01;
        if (current_nb01 == 0 || current_nb01 * (size_t)entry.ne01 != entry.current_nbytes) {
            std::fprintf(stderr, "bad current nb01 tensor=%s expert=%d current_nbytes=%zu ne01=%ld\n",
                    entry.tensor.c_str(), entry.expert_idx, entry.current_nbytes, (long)entry.ne01);
            ok_all = false;
            continue;
        }

        int packed_type = 0;
        int64_t ne00 = 0;
        int64_t ne01 = 0;
        size_t packed_nb01 = 0;
        size_t packed_nbytes = 0;
        if (!lookup_v2(entry.tensor.c_str(), entry.expert_idx, &packed_type, &ne00, &ne01, &packed_nb01, &packed_nbytes) ||
                packed_type != entry.packed_type || ne00 != entry.ne00 || ne01 != entry.ne01 ||
                packed_nb01 != entry.packed_nb01 || packed_nbytes != entry.packed_nbytes) {
            std::fprintf(stderr, "v2 lookup mismatch tensor=%s expert=%d\n", entry.tensor.c_str(), entry.expert_idx);
            ok_all = false;
            continue;
        }

        std::vector<uint8_t> current_bytes(entry.current_nbytes);
        std::vector<uint8_t> lower_bytes(entry.packed_nbytes);
        size_t nread = 0;
        int source_is_gguf = 0;
        if (!read_current(entry.tensor.c_str(), entry.expert_idx, entry.current_nbytes,
                    current_bytes.data(), current_bytes.size(), &nread, &source_is_gguf) ||
                nread != entry.current_nbytes) {
            std::fprintf(stderr, "current pack read failed tensor=%s expert=%d nread=%zu expected=%zu\n",
                    entry.tensor.c_str(), entry.expert_idx, nread, entry.current_nbytes);
            ok_all = false;
            continue;
        }
        if (!read_v2(entry.tensor.c_str(), entry.expert_idx, lower_bytes.data(), lower_bytes.size(), &nread) ||
                nread != entry.packed_nbytes) {
            std::fprintf(stderr, "v2 read failed tensor=%s expert=%d nread=%zu expected=%zu\n",
                    entry.tensor.c_str(), entry.expert_idx, nread, entry.packed_nbytes);
            ok_all = false;
            continue;
        }

        std::mt19937 rng((uint32_t)(0x5eed0000u ^ (uint32_t)entry.expert_idx ^ (uint32_t)entry.ne00));
        std::normal_distribution<float> dist(0.0f, 0.04f);
        metrics entry_metrics;
        for (int iter = 0; iter < activations_per_entry; ++iter) {
            std::vector<float> src1((size_t)entry.ne00);
            for (float & v : src1) {
                v = dist(rng);
            }
            std::vector<float> current_out;
            std::vector<float> lower_out;
            const bool current_ok = run_mmvq(mmvq, inv.current_type, current_bytes,
                    entry.ne00, entry.ne01, current_nb01, src1, current_out, stream);
            const bool lower_ok = run_mmvq(mmvq, entry.packed_type, lower_bytes,
                    entry.ne00, entry.ne01, entry.packed_nb01, src1, lower_out, stream);
            if (!current_ok || !lower_ok || current_out.size() != lower_out.size()) {
                std::fprintf(stderr, "mmvq compare failed tensor=%s expert=%d iter=%d\n",
                        entry.tensor.c_str(), entry.expert_idx, iter);
                ok_all = false;
                break;
            }
            add_metrics(entry_metrics, current_out, lower_out);
            add_metrics(total, current_out, lower_out);
        }
        if (entry_metrics.rows > 0) {
            std::printf("%s\t%d\t%s\t%s\t%s\t%zu\t%zu\t%.6f\t%d\t%.8f\t%.8f\t%.8f\t%.8g\t%.8g\t%.8g\t%d\n",
                    entry.tensor.c_str(),
                    entry.expert_idx,
                    entry.kind.c_str(),
                    inv.current_type_name.c_str(),
                    entry.remote_type.c_str(),
                    entry.current_nbytes,
                    entry.packed_nbytes,
                    (double)entry.packed_nbytes / (double)entry.current_nbytes,
                    entry_metrics.rows,
                    entry_metrics.rel_l2_sum / entry_metrics.rows,
                    entry_metrics.rel_l2_max,
                    entry_metrics.cosine_sum / entry_metrics.rows,
                    entry_metrics.mean_abs_sum / entry_metrics.rows,
                    entry_metrics.max_abs,
                    entry_metrics.current_l2_sum / entry_metrics.rows,
                    source_is_gguf);
        }
    }

    std::fprintf(stderr,
            "summary entries=%zu activation_rows=%d mean_rel_l2=%.8f max_rel_l2=%.8f mean_cosine=%.8f mean_abs_error=%.8g max_abs_error=%.8g mean_current_l2=%.8g\n",
            entries.size(),
            total.rows,
            total.rows ? total.rel_l2_sum / total.rows : 0.0,
            total.rel_l2_max,
            total.rows ? total.cosine_sum / total.rows : 0.0,
            total.rows ? total.mean_abs_sum / total.rows : 0.0,
            total.max_abs,
            total.rows ? total.current_l2_sum / total.rows : 0.0);

    cudaStreamDestroy(stream);
    dlclose(handle);
    return ok_all ? 0 : 1;
}
