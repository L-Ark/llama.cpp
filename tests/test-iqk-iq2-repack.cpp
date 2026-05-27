#include "ggml.h"

#include "../ggml/src/ggml-quants.h"
#include "../ggml/src/iqk/iqk_mul_mat.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(GGML_USE_CUDA)
extern "C" bool ggml_cuda_moe_iq2_q8k_r8_selftest(void);
#endif

static float value_for_prompt(int col) {
    const int a = col * 19 + (col / 7) * 3;
    const float wave = std::cos(0.017f * (float)(col + 5));
    return 0.11f * (float)((a % 29) - 14) + wave;
}

static void fill_iq2_row(uint8_t * dst, size_t stride, int row, int k) {
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
    (void)stride;
}

int main() {
    constexpr int k = 256 * 2;
    constexpr int n_rows = 8;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_quantize_init(GGML_TYPE_IQ2_S);
    ggml_quantize_init(GGML_TYPE_Q8_K);

    const ggml_type_traits_t q8_traits = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8_traits.from_float) {
        std::fprintf(stderr, "missing quantization callbacks\n");
        return 1;
    }

    const size_t iq2_stride = ggml_row_size(GGML_TYPE_IQ2_S, k);
    const size_t q8_stride = ggml_row_size(GGML_TYPE_Q8_K, k);
    const size_t r8_stride = ggml_row_size(GGML_TYPE_Q8_K_R8, k);

    std::vector<float> prompt(k);
    std::vector<uint8_t> iq2((size_t)n_rows * iq2_stride);
    std::vector<uint8_t> q8(q8_stride);
    std::vector<uint8_t> r8((size_t)n_rows * r8_stride);
    std::vector<float> direct(n_rows);
    std::vector<float> repacked(n_rows);
    std::vector<float> scalar(n_rows);
    std::vector<float> deq_row(k);
    std::vector<float> deq_prompt(k);

    for (int r = 0; r < n_rows; ++r) {
        fill_iq2_row(iq2.data() + (size_t)r * iq2_stride, iq2_stride, r, k);
    }
    const block_iq2_s * first_after_fill = (const block_iq2_s *)iq2.data();
    const float first_d_after_fill = ggml_fp16_to_fp32(first_after_fill->d);

    for (int c = 0; c < k; ++c) {
        prompt[c] = value_for_prompt(c);
    }
    q8_traits.from_float(prompt.data(), q8.data(), k);
    dequantize_row_q8_K((const block_q8_K *)q8.data(), deq_prompt.data(), k);
    float max_prompt_abs = 0.0f;
    for (int c = 0; c < k; ++c) {
        max_prompt_abs = std::max(max_prompt_abs, std::fabs(deq_prompt[c]));
    }

    float max_row_abs = 0.0f;
    for (int r = 0; r < n_rows; ++r) {
        dequantize_row_iq2_s((const block_iq2_s *)(iq2.data() + (size_t)r * iq2_stride), deq_row.data(), k);
        double sum = 0.0;
        for (int c = 0; c < k; ++c) {
            max_row_abs = std::max(max_row_abs, std::fabs(deq_row[c]));
            sum += (double)deq_row[c] * (double)deq_prompt[c];
        }
        scalar[r] = (float)sum;
    }

    if (!iqk_convert_repack_q8_r8(GGML_TYPE_IQ2_S, k, iq2.data(), iq2_stride, r8.data(), k, n_rows)) {
        std::fprintf(stderr, "iqk_convert_repack_q8_r8(IQ2_S -> Q8_K_R8) failed\n");
        return 1;
    }
    if (!iqk_mul_mat(n_rows, 1, k, GGML_TYPE_IQ2_S, iq2.data(), iq2_stride,
                GGML_TYPE_Q8_K, q8.data(), q8_stride, direct.data(), 0, 0, 1)) {
        std::fprintf(stderr, "direct IQ2_S x Q8_K multiply failed\n");
        return 1;
    }
    if (!iqk_mul_mat(n_rows, 1, k, GGML_TYPE_Q8_K_R8, r8.data(), r8_stride,
                GGML_TYPE_Q8_K, q8.data(), q8_stride, repacked.data(), 0, 0, 1)) {
        std::fprintf(stderr, "repacked Q8_K_R8 x Q8_K multiply failed\n");
        return 1;
    }

    float max_abs = 0.0f;
    float max_scalar_abs = 0.0f;
    float max_repack_delta = 0.0f;
    float max_repacked_abs = 0.0f;
    for (int r = 0; r < n_rows; ++r) {
        max_scalar_abs = std::max(max_scalar_abs, std::fabs(scalar[r]));
        const float direct_diff = std::fabs(direct[r] - scalar[r]);
        max_abs = std::max(max_abs, direct_diff);
        max_repack_delta = std::max(max_repack_delta, std::fabs(repacked[r] - direct[r]));
        max_repacked_abs = std::max(max_repacked_abs, std::fabs(repacked[r]));
        if (direct_diff > 1.0e-3f) {
            std::fprintf(stderr, "row %d direct mismatch scalar=%g direct=%g abs=%g\n",
                    r, scalar[r], direct[r], direct_diff);
            return 1;
        }
    }
    if (max_scalar_abs <= 1.0e-3f) {
        std::fprintf(stderr, "scalar oracle is unexpectedly zero: max_row_abs=%g max_prompt_abs=%g\n",
                max_row_abs, max_prompt_abs);
        std::fprintf(stderr, "strides: iq2=%zu q8=%zu r8=%zu\n", iq2_stride, q8_stride, r8_stride);
        const block_iq2_s * first = (const block_iq2_s *)iq2.data();
        std::fprintf(stderr, "first block: d_after_fill=%g d=%g qs0=%u qh0=%u scale0=%u\n",
                first_d_after_fill, ggml_fp16_to_fp32(first->d),
                (unsigned)first->qs[0], (unsigned)first->qh[0], (unsigned)first->scales[0]);
        return 1;
    }
    if (max_repacked_abs <= 1.0e-3f) {
        std::fprintf(stderr, "repacked oracle is unexpectedly zero\n");
        return 1;
    }

    std::printf("IQ2_S repack comparator ok: rows=%d k=%d direct_vs_scalar=%g repacked_vs_direct=%g max_repacked_abs=%g\n",
            n_rows, k, max_abs, max_repack_delta, max_repacked_abs);
#if defined(GGML_USE_CUDA)
    const char *cuda_compare = std::getenv("GGML_MOE_IQ2_CUDA_COMPARE");
    if (cuda_compare && cuda_compare[0] && cuda_compare[0] != '0') {
        if (!ggml_cuda_moe_iq2_q8k_r8_selftest()) {
            ggml_free(ctx);
            return 1;
        }
    }
#endif
    ggml_free(ctx);
    return 0;
}
