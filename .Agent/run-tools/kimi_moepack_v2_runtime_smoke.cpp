// Default-off smoke test for the CUDA MoE GGMLMOEPACKv2 debug reader.
//
// This tool intentionally uses dlopen/dlsym so it can validate an already-built
// libggml-cuda.so without adding a CMake target or loading a model.

#include <dlfcn.h>

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

struct expected_entry {
    const char * tensor;
    int expert_idx;
    int packed_type;
    int64_t ne00;
    int64_t ne01;
    size_t nb01;
    size_t nbytes;
};

static uint8_t expected_payload_byte(const expected_entry & entry, size_t i) {
    const int seed = entry.expert_idx * 17 + entry.packed_type * 31;
    return (uint8_t)((seed + (int)i * 13) & 0xff);
}

static bool check_entry(
        lookup_fn lookup,
        read_fn read,
        const expected_entry & expected,
        bool verify_synthetic_payload) {
    int packed_type = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
    size_t nbytes = 0;

    if (!lookup(expected.tensor, expected.expert_idx, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
        std::fprintf(stderr, "lookup failed tensor=%s expert=%d\n", expected.tensor, expected.expert_idx);
        return false;
    }
    if (packed_type != expected.packed_type || ne00 != expected.ne00 ||
            ne01 != expected.ne01 || nb01 != expected.nb01 || nbytes != expected.nbytes) {
        std::fprintf(stderr,
                "metadata mismatch tensor=%s expert=%d got type=%d ne00=%ld ne01=%ld nb01=%zu nbytes=%zu\n",
                expected.tensor, expected.expert_idx, packed_type, (long)ne00, (long)ne01, nb01, nbytes);
        return false;
    }

    std::vector<uint8_t> payload(expected.nbytes);
    size_t nread = 0;
    if (!read(expected.tensor, expected.expert_idx, payload.data(), payload.size(), &nread) ||
            nread != expected.nbytes) {
        std::fprintf(stderr,
                "read failed tensor=%s expert=%d nread=%zu expected=%zu\n",
                expected.tensor, expected.expert_idx, nread, expected.nbytes);
        return false;
    }
    if (verify_synthetic_payload) {
        for (size_t i = 0; i < payload.size(); ++i) {
            const uint8_t want = expected_payload_byte(expected, i);
            if (payload[i] != want) {
                std::fprintf(stderr,
                        "payload mismatch tensor=%s expert=%d off=%zu got=%u want=%u\n",
                        expected.tensor, expected.expert_idx, i, (unsigned)payload[i], (unsigned)want);
                return false;
            }
        }
    }
    return true;
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

static bool parse_manifest_entry(
        const std::vector<std::string> & header,
        const std::vector<std::string> & row,
        expected_entry & out) {
    std::unordered_map<std::string, size_t> cols;
    for (size_t i = 0; i < header.size(); ++i) {
        cols[header[i]] = i;
    }
    auto get = [&](const char * name) -> const std::string * {
        auto it = cols.find(name);
        if (it == cols.end() || it->second >= row.size()) {
            return nullptr;
        }
        return &row[it->second];
    };

    const std::string * tensor = get("tensor");
    const std::string * expert_idx = get("expert_idx");
    const std::string * packed_type = get("packed_type");
    const std::string * packed_nbytes = get("packed_nbytes");
    const std::string * packed_ne00 = get("packed_ne00");
    const std::string * packed_ne01 = get("packed_ne01");
    const std::string * packed_nb01 = get("packed_nb01");
    if (!tensor || !expert_idx || !packed_type || !packed_nbytes ||
            !packed_ne00 || !packed_ne01 || !packed_nb01) {
        return false;
    }

    out.tensor = nullptr;
    out.expert_idx = std::stoi(*expert_idx);
    out.packed_type = std::stoi(*packed_type);
    out.ne00 = std::stoll(*packed_ne00);
    out.ne01 = std::stoll(*packed_ne01);
    out.nb01 = (size_t)std::stoull(*packed_nb01);
    out.nbytes = (size_t)std::stoull(*packed_nbytes);
    return !tensor->empty() && out.expert_idx >= 0 && out.packed_type > 0 &&
        out.ne00 > 0 && out.ne01 > 0 && out.nb01 > 0 && out.nbytes > 0;
}

static bool check_manifest(lookup_fn lookup, read_fn read, const char * manifest_path) {
    std::ifstream in(manifest_path);
    if (!in) {
        std::fprintf(stderr, "manifest open failed: %s\n", manifest_path);
        return false;
    }

    std::string line;
    if (!std::getline(in, line)) {
        std::fprintf(stderr, "manifest empty: %s\n", manifest_path);
        return false;
    }
    const std::vector<std::string> header = split_tsv_line(line);

    size_t rows = 0;
    std::vector<std::string> tensor_storage;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = split_tsv_line(line);
        expected_entry entry = {};
        if (!parse_manifest_entry(header, row, entry)) {
            std::fprintf(stderr, "manifest row parse failed line=%zu\n", rows + 2);
            return false;
        }

        std::unordered_map<std::string, size_t> cols;
        for (size_t i = 0; i < header.size(); ++i) {
            cols[header[i]] = i;
        }
        const auto tensor_col = cols.find("tensor");
        if (tensor_col == cols.end() || tensor_col->second >= row.size()) {
            std::fprintf(stderr, "manifest tensor column missing line=%zu\n", rows + 2);
            return false;
        }
        tensor_storage.push_back(row[tensor_col->second]);
        entry.tensor = tensor_storage.back().c_str();

        if (!check_entry(lookup, read, entry, false)) {
            return false;
        }
        ++rows;
    }

    if (rows == 0) {
        std::fprintf(stderr, "manifest has no data rows: %s\n", manifest_path);
        return false;
    }
    std::printf("manifest_smoke pass rows=%zu manifest=%s\n", rows, manifest_path);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s LIBGGML_CUDA_SO V2_EXPERT_PACK [MANIFEST_TSV]\n", argv[0]);
        return 2;
    }

    const char * lib_path = argv[1];
    const char * pack_path = argv[2];
    if (setenv("GGML_MOE_EXPERT_PACK_V2", pack_path, 1) != 0) {
        std::perror("setenv GGML_MOE_EXPERT_PACK_V2");
        return 1;
    }

    void * handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto lookup = (lookup_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_lookup_debug");
    auto read = (read_fn)dlsym(handle, "ggml_cuda_moe_expert_pack_v2_read_debug");
    if (!lookup || !read) {
        std::fprintf(stderr, "required v2 debug symbols missing\n");
        dlclose(handle);
        return 1;
    }

    size_t checked = 0;
    if (argc == 4) {
        if (!check_manifest(lookup, read, argv[3])) {
            dlclose(handle);
            return 1;
        }
        std::ifstream count_in(argv[3]);
        std::string count_line;
        if (std::getline(count_in, count_line)) {
            while (std::getline(count_in, count_line)) {
                if (!count_line.empty()) {
                    ++checked;
                }
            }
        }
    } else {
        const expected_entry entries[] = {
            {"blk.1.ffn_up_exps.weight", 7, 24, 256, 4, 16, 64},
            {"blk.1.ffn_down_exps.weight", 11, 10, 4, 256, 32, 128},
        };
        for (const expected_entry & entry : entries) {
            if (!check_entry(lookup, read, entry, true)) {
                dlclose(handle);
                return 1;
            }
        }
        checked = sizeof(entries) / sizeof(entries[0]);

        int packed_type = 0;
        int64_t ne00 = 0;
        int64_t ne01 = 0;
        size_t nb01 = 0;
        size_t nbytes = 0;
        if (lookup("blk.1.ffn_up_exps.weight", 12345, &packed_type, &ne00, &ne01, &nb01, &nbytes)) {
            std::fprintf(stderr, "negative lookup unexpectedly succeeded\n");
            dlclose(handle);
            return 1;
        }
    }

    std::printf("kimi_moepack_v2_runtime_smoke pass entries=%zu pack=%s lib=%s\n",
            checked, pack_path, lib_path);
    dlclose(handle);
    return 0;
}
