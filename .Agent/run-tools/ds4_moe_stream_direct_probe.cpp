#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_stream_row_mapping_probe;

extern "C" bool ggml_cuda_moe_stream_batch(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float * src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_q8_0_ne1,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const int64_t * matrix_row_counts,
    const ggml_moe_stream_row_mapping_probe * rows,
    int64_t rows_stride);

static void fill_pattern(std::vector<float> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        const int centered = (int) (i % 17) - 8;
        data[i] = 0.125f + 0.01f * (float) centered;
    }
}

static void compute_reference(
        ggml_type type,
        const std::vector<uint8_t> & src0_q,
        size_t nb01,
        size_t nb02,
        int64_t ne01,
        int64_t ne00,
        const std::vector<float> & src1_f32,
        std::vector<float> & ref) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> row((size_t) ne00);
    for (int64_t col = 0; col < ne01; ++col) {
        const void * qrow = src0_q.data() + (size_t) col * nb01;
        traits->to_float(qrow, row.data(), ne00);
        double acc = 0.0;
        for (int64_t k = 0; k < ne00; ++k) {
            acc += (double) row[(size_t) k] * (double) src1_f32[(size_t) k];
        }
        ref[(size_t) col] = (float) acc;
    }
    (void) nb02;
}

static bool run_decline_case(ggml_type type, const char * type_name) {
    const int64_t n_as = 1;
    const int64_t ne01 = 16;
    const int64_t ne00 = 256;
    const size_t nb01 = ggml_row_size(type, ne00);
    const size_t nb02 = nb01 * (size_t) ne01;

    std::vector<float> src0_f32((size_t) n_as * (size_t) ne01 * (size_t) ne00);
    std::vector<float> src1_f32((size_t) ne00);
    std::vector<float> dst((size_t) ne01, 0.0f);
    std::vector<float> ref((size_t) ne01, 0.0f);
    std::vector<uint8_t> src0_q((size_t) n_as * nb02);
    std::vector<float> imatrix((size_t) ne00, 1.0f);
    int64_t counts[1] = {1};
    ggml_moe_stream_row_mapping_probe rows[1] = {{0, 0}};

    fill_pattern(src0_f32);
    fill_pattern(src1_f32);
    ggml_quantize_chunk(type, src0_f32.data(), src0_q.data(), 0, n_as * ne01, ne00, imatrix.data());
    compute_reference(type, src0_q, nb01, nb02, ne01, ne00, src1_f32, ref);

    const bool done = ggml_cuda_moe_stream_batch(
        (int) type,
        "blk.0.ffn_down_exps.weight",
        src0_q.data(),
        n_as,
        ne01,
        ne00,
        nb01,
        nb02,
        src1_f32.data(),
        ne00 * sizeof(float),
        ne00 * sizeof(float),
        nullptr,
        0,
        0,
        dst.data(),
        sizeof(float),
        ne01 * sizeof(float),
        counts,
        rows,
        1);

    const bool expect_accept = std::getenv("GGML_MOE_STREAM_DOWN_LOWBIT_PROBE") != nullptr;
    if (done) {
        float max_abs = 0.0f;
        double mean_abs = 0.0;
        bool finite = true;
        for (int64_t i = 0; i < ne01; ++i) {
            finite = finite && std::isfinite(dst[(size_t) i]) && std::isfinite(ref[(size_t) i]);
            const float err = std::fabs(dst[(size_t) i] - ref[(size_t) i]);
            finite = finite && std::isfinite(err);
            if (max_abs < err) max_abs = err;
            mean_abs += err;
        }
        mean_abs /= (double) ne01;
        std::fprintf(stdout, "accept type=%s nb01=%zu nb02=%zu finite=%d max_abs=%.9g mean_abs=%.9g first_gpu=%.9g first_ref=%.9g\n",
                type_name, nb01, nb02, finite ? 1 : 0, max_abs, mean_abs, dst[0], ref[0]);
        return expect_accept && finite && max_abs < 0.25f;
    }

    if (expect_accept) {
        std::fprintf(stderr, "unexpected stream decline for %s\n", type_name);
        return false;
    }
    std::fprintf(stdout, "decline-ok type=%s nb01=%zu nb02=%zu\n", type_name, nb01, nb02);
    return true;
}

static long parse_env_long(const char * name, long fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    return std::strtol(value, nullptr, 10);
}

static bool run_real_rows_case() {
    const char * path = std::getenv("GGML_MOE_STREAM_REAL_ROWS_BIN");
    if (!path || !path[0]) {
        return true;
    }

    const ggml_type type = (ggml_type) parse_env_long("GGML_MOE_STREAM_REAL_ROWS_TYPE", (long) GGML_TYPE_IQ3_S);
    const char * type_name = ggml_type_name(type);
    const int64_t ne00 = parse_env_long("GGML_MOE_STREAM_REAL_ROWS_NE00", 2048);
    const int64_t ne01 = parse_env_long("GGML_MOE_STREAM_REAL_ROWS_NE01", 16);
    const size_t nb01 = (size_t) parse_env_long("GGML_MOE_STREAM_REAL_ROWS_NB01", (long) ggml_row_size(type, ne00));
    const size_t nb02 = nb01 * (size_t) ne01;
    const size_t bytes = nb01 * (size_t) ne01;

    std::vector<uint8_t> src0_q(bytes);
    std::ifstream in(path, std::ios::binary);
    if (!in.read((char *) src0_q.data(), (std::streamsize) src0_q.size())) {
        std::fprintf(stderr, "failed to read real rows path=%s expected_bytes=%zu got=%lld\n",
                path, src0_q.size(), (long long) in.gcount());
        return false;
    }

    std::vector<float> src1_f32((size_t) ne00);
    std::vector<float> dst((size_t) ne01, 0.0f);
    std::vector<float> ref((size_t) ne01, 0.0f);
    int64_t counts[1] = {1};
    ggml_moe_stream_row_mapping_probe rows[1] = {{0, 0}};

    fill_pattern(src1_f32);
    compute_reference(type, src0_q, nb01, nb02, ne01, ne00, src1_f32, ref);

    const bool done = ggml_cuda_moe_stream_batch(
        (int) type,
        "blk.0.ffn_down_exps.weight",
        src0_q.data(),
        1,
        ne01,
        ne00,
        nb01,
        nb02,
        src1_f32.data(),
        (size_t) ne00 * sizeof(float),
        (size_t) ne00 * sizeof(float),
        nullptr,
        0,
        0,
        dst.data(),
        sizeof(float),
        (size_t) ne01 * sizeof(float),
        counts,
        rows,
        1);

    if (!done) {
        std::fprintf(stderr, "real-rows unexpected stream decline type=%s path=%s\n", type_name, path);
        return false;
    }

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    bool finite = true;
    for (int64_t i = 0; i < ne01; ++i) {
        finite = finite && std::isfinite(dst[(size_t) i]) && std::isfinite(ref[(size_t) i]);
        const float err = std::fabs(dst[(size_t) i] - ref[(size_t) i]);
        finite = finite && std::isfinite(err);
        if (max_abs < err) max_abs = err;
        mean_abs += err;
    }
    mean_abs /= (double) ne01;
    std::fprintf(stdout, "real-rows accept type=%s ne00=%ld ne01=%ld nb01=%zu finite=%d max_abs=%.9g mean_abs=%.9g first_gpu=%.9g first_ref=%.9g\n",
            type_name, (long) ne00, (long) ne01, nb01, finite ? 1 : 0, max_abs, mean_abs, dst[0], ref[0]);
    return finite && max_abs < 0.25f;
}

int main() {
    setenv("GGML_MOE_STREAM_DECLINE_DEBUG", "1", 1);
    ggml_cpu_init();

    bool ok = true;
    ok = run_decline_case(GGML_TYPE_IQ1_S, "iq1_s") && ok;
    ok = run_decline_case(GGML_TYPE_IQ1_M, "iq1_m") && ok;
    ok = run_decline_case(GGML_TYPE_Q2_K, "q2_K") && ok;
    ok = run_real_rows_case() && ok;

    return ok ? 0 : 1;
}
