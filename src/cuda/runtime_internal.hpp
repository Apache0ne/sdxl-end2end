#pragma once

#include "sdxl/cuda/runtime.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sdxl::cuda {

struct MatmulKey final {
    std::uint64_t m = 0;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    int data_type = 0;

    bool operator==(const MatmulKey&) const noexcept = default;
};

struct MatmulKeyHash final {
    std::size_t operator()(const MatmulKey& value) const noexcept {
        std::size_t hash = static_cast<std::size_t>(value.m);
        hash ^= static_cast<std::size_t>(value.n) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::size_t>(value.k) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::size_t>(value.data_type) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

struct MatmulDescriptors final {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~MatmulDescriptors();
    MatmulDescriptors() = default;
    MatmulDescriptors(const MatmulDescriptors&) = delete;
    MatmulDescriptors& operator=(const MatmulDescriptors&) = delete;
};

struct MatmulPlan final {
    cublasLtMatmulAlgo_t algorithm{};
    std::size_t workspace_bytes = 0;
    std::shared_ptr<MatmulDescriptors> descriptors;
};

struct ConvolutionKey final {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    int k = 0;
    int r = 0;
    int s = 0;
    int stride_y = 0;
    int stride_x = 0;
    int pad_y = 0;
    int pad_x = 0;
    int data_type = 0;

    bool operator==(const ConvolutionKey&) const noexcept = default;
};

struct ConvolutionKeyHash final {
    std::size_t operator()(const ConvolutionKey& value) const noexcept {
        const int fields[] = {value.n, value.c, value.h, value.w, value.k, value.r,
                              value.s, value.stride_y, value.stride_x, value.pad_y, value.pad_x,
                              value.data_type};
        std::size_t hash = 0;
        for (const int field : fields) {
            hash ^= static_cast<std::size_t>(field) + 0x9e3779b97f4a7c15ULL +
                    (hash << 6U) + (hash >> 2U);
        }
        return hash;
    }
};

struct ConvolutionDescriptors final {
    cudnnTensorDescriptor_t input = nullptr;
    cudnnTensorDescriptor_t output = nullptr;
    cudnnTensorDescriptor_t bias = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t convolution = nullptr;

    ConvolutionDescriptors() = default;
    ~ConvolutionDescriptors();
    ConvolutionDescriptors(const ConvolutionDescriptors&) = delete;
    ConvolutionDescriptors& operator=(const ConvolutionDescriptors&) = delete;
};

struct ConvolutionPlan final {
    cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    std::size_t workspace_bytes = 0;
    int output_n = 0;
    int output_c = 0;
    int output_h = 0;
    int output_w = 0;
    std::shared_ptr<ConvolutionDescriptors> descriptors;
};

struct RuntimeState final {
    RuntimeOptions options;
    int device = 0;
    cudaDeviceProp properties{};
    cudaStream_t stream = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    cudnnHandle_t cudnn = nullptr;
    curandGenerator_t curand = nullptr;
    cudaMemPool_t memory_pool = nullptr;
    void* cublas_workspace = nullptr;
    void* cudnn_workspace = nullptr;
    std::size_t cudnn_workspace_bytes = 0;
    std::mutex cublas_mutex;
    std::mutex cudnn_mutex;
    std::mutex curand_mutex;
    std::mutex sdpa_mutex;
    mutable std::mutex arena_mutex;
    std::multimap<std::size_t, void*> arena_free_blocks;
    void* arena_slab = nullptr;
    std::size_t arena_slab_bytes = 0;
    // offset -> free byte count. Adjacent ranges are coalesced on release.
    std::map<std::size_t, std::size_t> arena_slab_free_ranges;
    std::unordered_map<void*, std::pair<std::size_t, std::size_t>> arena_slab_live_ranges;
    std::size_t arena_live_bytes = 0;
    std::size_t arena_cached_bytes = 0;
    std::size_t arena_peak_live_bytes = 0;
    std::size_t arena_driver_allocations = 0;
    std::size_t arena_cache_hits = 0;
    std::size_t arena_slab_suballocations = 0;
    std::size_t persistent_live_bytes = 0;
    std::size_t persistent_allocations = 0;
    bool graph_allocation_capture_active = false;
    std::unordered_map<MatmulKey, MatmulPlan, MatmulKeyHash> matmul_plans;

    std::atomic<std::size_t> int8_linear_calls{0};
    std::atomic<std::size_t> int8_cublaslt_imma_calls{0};
    std::atomic<std::size_t> int8_dp4a_fallback_calls{0};
    std::atomic<std::size_t> int8_tensor_core_plan_misses{0};
    std::atomic<std::size_t> int8_tensor_core_execution_failures{0};

    std::unordered_map<ConvolutionKey, ConvolutionPlan, ConvolutionKeyHash> convolution_plans;
    std::unordered_map<std::string, std::shared_ptr<void>> sdpa_plans;

    [[nodiscard]] void* acquire_block(std::size_t bytes);
    void release_block(void* pointer, std::size_t bytes) noexcept;
    [[nodiscard]] void* acquire_persistent_block(std::size_t bytes);
    void release_persistent_block(void* pointer, std::size_t bytes) noexcept;
    void trim_arena() noexcept;
    [[nodiscard]] MemoryArenaStats arena_stats() const noexcept;
    [[nodiscard]] void* ensure_cudnn_workspace(std::size_t bytes);
    void begin_graph_allocation_capture();
    void end_graph_allocation_capture() noexcept;

    ~RuntimeState();
};

} // namespace sdxl::cuda
