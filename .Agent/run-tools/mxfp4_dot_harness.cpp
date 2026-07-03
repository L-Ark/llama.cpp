#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-cpu/quants.h"
#include "ggml-cpu/repack.h"

static block_mxfp4x8 make_mxfp4x8(const block_mxfp4 * in) {
    block_mxfp4x8 out{};
    for (int i = 0; i < 8; ++i) {
        out.e[i] = in[i].e;
    }
    const int end = QK_MXFP4 * 4 / 8;
    for (int i = 0; i < end; ++i) {
        const int src_id = i % 8;
        const int src_offset = (i / 8) * 8;
        const int dst_offset = i * 8;
        std::memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    }
    return out;
}

static void repack_mxfp4x8(const std::vector<block_mxfp4> & src, std::vector<block_mxfp4x8> & dst, int rows, int nb) {
    dst.resize((rows / 8) * nb);
    block_mxfp4 tmp[8];
    block_mxfp4x8 * d = dst.data();
    for (int r = 0; r < rows; r += 8) {
        for (int b = 0; b < nb; ++b) {
            for (int i = 0; i < 8; ++i) {
                tmp[i] = src[(r + i) * nb + b];
            }
            *d++ = make_mxfp4x8(tmp);
        }
    }
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void bench_shape(int k, int rows, int iters) {
    if (rows % 8 != 0 || k % QK_MXFP4 != 0) {
        std::fprintf(stderr, "invalid shape k=%d rows=%d\n", k, rows);
        std::exit(2);
    }

    const int nb = k / QK_MXFP4;
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> i8_dist(-64, 63);

    std::vector<block_mxfp4> x((size_t) rows * nb);
    std::vector<block_q8_0> y(nb);
    for (auto & b : x) {
        b.e = (uint8_t) (120 + (byte_dist(rng) % 16));
        for (auto & q : b.qs) {
            q = (uint8_t) byte_dist(rng);
        }
    }
    for (auto & b : y) {
        b.d = GGML_FP32_TO_FP16(0.01f + (byte_dist(rng) % 8) * 0.001f);
        for (auto & q : b.qs) {
            q = (int8_t) i8_dist(rng);
        }
    }

    std::vector<block_mxfp4x8> xr;
    const double repack_t0 = now_ms();
    repack_mxfp4x8(x, xr, rows, nb);
    const double repack_ms = now_ms() - repack_t0;

    std::vector<float> out_row(rows), out_repack(rows);
    for (int r = 0; r < rows; ++r) {
        ggml_vec_dot_mxfp4_q8_0(k, &out_row[r], 0, &x[(size_t) r * nb], 0, y.data(), 0, 1);
    }
    ggml_gemv_mxfp4_8x8_q8_0(k, out_repack.data(), 0, xr.data(), y.data(), 1, rows);

    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double d = std::abs((double) out_row[r] - (double) out_repack[r]);
        max_abs = std::max(max_abs, d);
        mean_abs += d;
    }
    mean_abs /= rows;

    volatile double sink = 0.0;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < rows; ++r) {
            float v;
            ggml_vec_dot_mxfp4_q8_0(k, &v, 0, &x[(size_t) r * nb], 0, y.data(), 0, 1);
            sink += std::fabs((double) v) + 1.0e-12;
        }
    }
    const double row_ms = now_ms() - t0;

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        ggml_gemv_mxfp4_8x8_q8_0(k, out_repack.data(), 0, xr.data(), y.data(), 1, rows);
        sink += std::fabs((double) out_repack[it % rows]) + 1.0e-12;
    }
    const double repack_gemv_ms = now_ms() - t0;

    std::printf(
        "shape k=%d rows=%d iters=%d repack_ms=%.3f row_ms=%.3f repack_gemv_ms=%.3f speedup=%.3f max_abs=%.9g mean_abs=%.9g sink=%.9g\n",
        k, rows, iters, repack_ms, row_ms, repack_gemv_ms, row_ms / repack_gemv_ms, max_abs, mean_abs, (double) sink);
}

int main(int argc, char ** argv) {
    int iters = 200;
    if (argc == 2) {
        iters = std::max(1, std::atoi(argv[1]));
    }

    bench_shape(4096, 2048, iters);
    bench_shape(2048, 4096, iters);
    return 0;
}
