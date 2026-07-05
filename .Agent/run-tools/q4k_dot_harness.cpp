#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "ggml-cpu/quants.h"

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void bench_shape(int k, int rows, int iters, int nth) {
    if (k % QK_K != 0 || rows <= 0 || iters <= 0 || nth <= 0) {
        std::fprintf(stderr, "invalid shape k=%d rows=%d iters=%d threads=%d\n", k, rows, iters, nth);
        std::exit(2);
    }

    const int nb = k / QK_K;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> xf((size_t) rows * k);
    std::vector<float> yf(k);
    for (float & v : xf) {
        v = dist(rng);
    }
    for (float & v : yf) {
        v = dist(rng);
    }

    std::vector<block_q4_K> xq((size_t) rows * nb);
    std::vector<block_q8_K> yq(nb);

    const double q0 = now_ms();
    for (int r = 0; r < rows; ++r) {
        quantize_row_q4_K(&xf[(size_t) r * k], &xq[(size_t) r * nb], k);
    }
    quantize_row_q8_K(yf.data(), yq.data(), k);
    const double quant_ms = now_ms() - q0;

    std::vector<float> out(rows);
    for (int r = 0; r < rows; ++r) {
        ggml_vec_dot_q4_K_q8_K(k, &out[r], 0, &xq[(size_t) r * nb], 0, yq.data(), 0, 1);
    }

    std::vector<double> sinks(nth, 0.0);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(nth);

    const int chunk = (rows + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
        threads.emplace_back([&, t]() {
            const int r0 = std::min(rows, t * chunk);
            const int r1 = std::min(rows, r0 + chunk);
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            double sink = 0.0;
            for (int it = 0; it < iters; ++it) {
                for (int r = r0; r < r1; ++r) {
                    float v = 0.0f;
                    ggml_vec_dot_q4_K_q8_K(k, &v, 0, &xq[(size_t) r * nb], 0, yq.data(), 0, 1);
                    sink += std::fabs((double) v) + 1.0e-12;
                }
            }
            sinks[t] = sink;
        });
    }

    while (ready.load(std::memory_order_acquire) < nth) {
        std::this_thread::yield();
    }
    const double t0 = now_ms();
    go.store(true, std::memory_order_release);
    for (auto & th : threads) {
        th.join();
    }
    const double dot_ms = now_ms() - t0;

    double sink = 0.0;
    for (double v : sinks) {
        sink += v;
    }

    const uint64_t dot_calls = (uint64_t) rows * (uint64_t) iters;
    const uint64_t q4_src_bytes = (uint64_t) rows * (uint64_t) nb * (uint64_t) sizeof(block_q4_K) * (uint64_t) iters;
    const uint64_t q8_bytes = (uint64_t) nb * (uint64_t) sizeof(block_q8_K) * (uint64_t) iters;
    const double src_gib_s = (double) q4_src_bytes / (1024.0 * 1024.0 * 1024.0) / (dot_ms / 1000.0);
    const double total_gib_s = (double) (q4_src_bytes + q8_bytes) / (1024.0 * 1024.0 * 1024.0) / (dot_ms / 1000.0);
    const double us_per_dot = dot_ms * 1000.0 / (double) dot_calls;

    std::printf(
        "shape k=%d rows=%d iters=%d threads=%d quant_ms=%.3f dot_ms=%.3f dot_calls=%llu q4_src_bytes=%llu q8_bytes=%llu src_gib_s=%.3f total_gib_s=%.3f us_per_dot=%.6f sink=%.9g\n",
        k, rows, iters, nth, quant_ms, dot_ms,
        (unsigned long long) dot_calls,
        (unsigned long long) q4_src_bytes,
        (unsigned long long) q8_bytes,
        src_gib_s, total_gib_s, us_per_dot, sink);
}

int main(int argc, char ** argv) {
    int iters = 400;
    if (argc >= 2) {
        iters = std::max(1, std::atoi(argv[1]));
    }

    std::vector<int> thread_counts = {1, 8, 20, 24, 32};
    if (argc >= 3) {
        thread_counts.clear();
        for (int i = 2; i < argc; ++i) {
            thread_counts.push_back(std::max(1, std::atoi(argv[i])));
        }
    }

    for (int nth : thread_counts) {
        bench_shape(4096, 2048, iters, nth);
        bench_shape(2048, 4096, iters, nth);
    }
    return 0;
}
