#include "sdxl/cuda/ops.hpp"
#include "sdxl/cuda/runtime.hpp"
#include "sdxl/cuda/tensor.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

namespace {

sdxl::FloatTensor make_input(std::size_t batch,
                             std::size_t sequence,
                             std::size_t width,
                             float frequency) {
    sdxl::FloatTensor tensor;
    tensor.shape = {batch, sequence, width};
    tensor.values.resize(batch * sequence * width);
    for (std::size_t index = 0; index < tensor.values.size(); ++index) {
        tensor.values[index] = std::sin(static_cast<float>(index) * frequency) * 0.25F;
    }
    return tensor;
}

struct BenchmarkResult {
    double milliseconds = 0.0;
    sdxl::FloatTensor output;
};

struct ErrorMetrics {
    double max_absolute = 0.0;
    double root_mean_square = 0.0;
};

[[nodiscard]] ErrorMetrics compare_outputs(const sdxl::FloatTensor& candidate,
                                           const sdxl::FloatTensor& reference) {
    if (candidate.shape != reference.shape || candidate.values.size() != reference.values.size()) {
        throw std::runtime_error("attention benchmark output shape mismatch");
    }
    double squared_sum = 0.0;
    double max_absolute = 0.0;
    for (std::size_t index = 0; index < candidate.values.size(); ++index) {
        const double difference = static_cast<double>(candidate.values[index]) -
                                  static_cast<double>(reference.values[index]);
        max_absolute = std::max(max_absolute, std::abs(difference));
        squared_sum += difference * difference;
    }
    const double rms = candidate.values.empty()
        ? 0.0 : std::sqrt(squared_sum / static_cast<double>(candidate.values.size()));
    return ErrorMetrics{max_absolute, rms};
}

BenchmarkResult benchmark(sdxl::cuda::AttentionBackend backend,
                          const sdxl::FloatTensor& q,
                          const sdxl::FloatTensor& k,
                          const sdxl::FloatTensor& v,
                          std::size_t heads,
                          int warmup,
                          int iterations) {
    sdxl::cuda::RuntimeOptions options;
    options.attention_backend = backend;
    options.arena_reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    sdxl::cuda::Runtime runtime(options);
    sdxl::cuda::Ops ops(runtime);
    auto qd = sdxl::cuda::Tensor::from_host_f32(runtime, q);
    auto kd = sdxl::cuda::Tensor::from_host_f32(runtime, k);
    auto vd = sdxl::cuda::Tensor::from_host_f32(runtime, v);
    for (int iteration = 0; iteration < warmup; ++iteration) {
        auto output = ops.attention(qd, kd, vd, heads, false);
        (void)output;
    }
    runtime.synchronize();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    SDXL_CUDA_CHECK(cudaEventCreate(&start));
    SDXL_CUDA_CHECK(cudaEventCreate(&stop));
    SDXL_CUDA_CHECK(cudaEventRecord(start, runtime.stream()));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto output = ops.attention(qd, kd, vd, heads, false);
        (void)output;
    }
    SDXL_CUDA_CHECK(cudaEventRecord(stop, runtime.stream()));
    SDXL_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0F;
    SDXL_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    auto checked_output = ops.attention(qd, kd, vd, heads, false).to_host_f32(runtime);
    return BenchmarkResult{
        static_cast<double>(elapsed) / static_cast<double>(iterations),
        std::move(checked_output)};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t query_sequence = argc > 1 ? std::stoull(argv[1]) : 4096;
        const std::size_t key_sequence = argc > 2 ? std::stoull(argv[2]) : query_sequence;
        const std::size_t heads = argc > 3 ? std::stoull(argv[3]) : 10;
        const std::size_t batch = argc > 4 ? std::stoull(argv[4]) : 2;
        const int iterations = argc > 5 ? std::stoi(argv[5]) : 10;
        const std::size_t width = heads * 64;
        const auto q = make_input(batch, query_sequence, width, 0.0013F);
        const auto k = make_input(batch, key_sequence, width, 0.0017F);
        const auto v = make_input(batch, key_sequence, width, 0.0007F);
        const BenchmarkResult flash = benchmark(
            sdxl::cuda::AttentionBackend::FlashSM80, q, k, v, heads, 2, iterations);
        const BenchmarkResult warp = benchmark(
            sdxl::cuda::AttentionBackend::WarpOnline, q, k, v, heads, 1,
            std::max(1, std::min(iterations, 3)));
        const ErrorMetrics flash_error = compare_outputs(flash.output, warp.output);
        std::cout << std::fixed << std::setprecision(3)
                  << "shape B=" << batch << " Q=" << query_sequence << " K=" << key_sequence
                  << " H=" << heads << " D=64\n"
                  << "flash-sm80: " << flash.milliseconds << " ms\n"
                  << "warp-online: " << warp.milliseconds << " ms\n"
                  << "flash vs warp: " << (warp.milliseconds / flash.milliseconds) << "x\n"
                  << "flash error: max_abs=" << flash_error.max_absolute
                  << " rms=" << flash_error.root_mean_square << "\n";
        try {
            const BenchmarkResult cudnn = benchmark(
                sdxl::cuda::AttentionBackend::CuDnnSDPA, q, k, v, heads, 2, iterations);
            const ErrorMetrics cudnn_error = compare_outputs(cudnn.output, warp.output);
            std::cout << "cudnn-sdpa: " << cudnn.milliseconds << " ms\n"
                      << "cudnn vs warp: " << (warp.milliseconds / cudnn.milliseconds) << "x\n"
                      << "cudnn error: max_abs=" << cudnn_error.max_absolute
                      << " rms=" << cudnn_error.root_mean_square << "\n"
                      << "winner: " << (cudnn.milliseconds < flash.milliseconds
                          ? "cudnn-sdpa" : "flash-sm80") << "\n";
        } catch (const std::exception& unavailable) {
            std::cout << "cudnn-sdpa: unavailable (" << unavailable.what() << ")\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
