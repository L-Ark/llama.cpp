#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        data[i] = 0.125f + 0.01f * (float)((i % 17) - 8);
    }
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
    std::vector<uint8_t> src0_q((size_t) n_as * nb02);
    std::vector<float> imatrix((size_t) ne00, 1.0f);
    int64_t counts[1] = {1};
    ggml_moe_stream_row_mapping_probe rows[1] = {{0, 0}};

    fill_pattern(src0_f32);
    fill_pattern(src1_f32);
    ggml_quantize_chunk(type, src0_f32.data(), src0_q.data(), 0, n_as * ne01, ne00, imatrix.data());

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

    if (done) {
        std::fprintf(stderr, "unexpected stream accept for %s\n", type_name);
        return false;
    }

    std::fprintf(stdout, "decline-ok type=%s nb01=%zu nb02=%zu\n", type_name, nb01, nb02);
    return true;
}

int main() {
    setenv("GGML_MOE_STREAM_DECLINE_DEBUG", "1", 1);
    ggml_cpu_init();

    bool ok = true;
    ok = run_decline_case(GGML_TYPE_IQ1_S, "iq1_s") && ok;
    ok = run_decline_case(GGML_TYPE_IQ1_M, "iq1_m") && ok;
    ok = run_decline_case(GGML_TYPE_Q2_K, "q2_K") && ok;

    return ok ? 0 : 1;
}
