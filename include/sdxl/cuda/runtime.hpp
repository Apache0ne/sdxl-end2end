#pragma once

#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <curand.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sdxl::cuda {

class CudaError final : public std::runtime_error {
public:
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

void check_cuda(cudaError_t status, std::string_view expression, std::string_view file, int line);
void check_cublas(cublasStatus_t status, std::string_view expression, std::string_view file, int line);
void check_cudnn(cudnnStatus_t status, std::string_view expression, std::string_view file, int line);
void check_curand(curandStatus_t status, std::string_view expression, std::string_view file, int line);

#define SDXL_CUDA_CHECK(expr) ::sdxl::cuda::check_cuda((expr), #expr, __FILE__, __LINE__)
#define SDXL_CUBLAS_CHECK(expr) ::sdxl::cuda::check_cublas((expr), #expr, __FILE__, __LINE__)
#define SDXL_CUDNN_CHECK(expr) ::sdxl::cuda::check_cudnn((expr), #expr, __FILE__, __LINE__)
#define SDXL_CURAND_CHECK(expr) ::sdxl::cuda::check_curand((expr), #expr, __FILE__, __LINE__)

enum class FP8ExecutionMode : std::uint8_t {
    Unsupported,
    WeightOnlyTensorCore,
    NativeTensorCore
};

enum class FP8BackendPreference : std::uint8_t {
    Auto,
    NativeOnly,
    WeightOnly
};

enum class NonVAEAccumulation : std::uint8_t {
    Float16,
    Float32
};

// Attention execution policy. FlashSM80 is an in-tree, forward-only,
// tiled Tensor Core implementation specialized for SDXL head dimension 64.
// WarpOnline is the original scalar online-softmax reference fallback.
enum class AttentionBackend : std::uint8_t {
    Auto,
    CuDnnSDPA,
    FlashSM80,
    WarpOnline
};

[[nodiscard]] AttentionBackend parse_attention_backend(std::string_view value);
[[nodiscard]] const char* attention_backend_name(AttentionBackend backend) noexcept;

struct RuntimeOptions {
    int device = 0;
    std::size_t cublas_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
    std::size_t cudnn_workspace_limit_bytes = 512ULL * 1024ULL * 1024ULL;

    // TF32 is only used by explicitly tagged FP32 VAE tensors.
    bool allow_tf32 = true;
    bool deterministic = false;
    bool release_threshold_zero = false;

    // Reserve one persistent device slab and suballocate temporary tensors from it.
    // This removes cudaMalloc/cudaFree calls from the warmed denoising loop. Zero
    // disables the slab and retains exact-size fallback caching.
    std::size_t arena_reserve_bytes = 0;

    // Reuse fallback allocations that do not fit in the slab. Set to zero to
    // disable fallback caching.
    std::size_t arena_cache_limit_bytes = std::numeric_limits<std::size_t>::max();

    // Reject every persistent FP32 tensor except VAE tensors, FP8 scale metadata,
    // and explicit host/debug interop buffers.
    bool strict_non_vae_fp32 = true;

    // FP16 is the default for CLIP and non-FP8 UNet GEMMs/convolutions. Native
    // FP8 cuBLASLt still requires CUBLAS_COMPUTE_32F by NVIDIA contract, but no
    // FP32 model tensor is materialized.
    NonVAEAccumulation non_vae_accumulation = NonVAEAccumulation::Float16;
    FP8BackendPreference fp8_backend = FP8BackendPreference::Auto;

    // Auto prefers cuDNN Frontend SDPA when that optional header-only backend
    // was available at build time, otherwise the in-tree tiled SM80 path is
    // used for large FP16 head-dim-64 buckets. Warp-online and FP32 VAE
    // attention remain correctness fallbacks.
    AttentionBackend attention_backend = AttentionBackend::Auto;
};

struct RuntimeState;

struct MemoryArenaStats {
    std::size_t live_bytes = 0;
    std::size_t cached_bytes = 0;
    std::size_t peak_live_bytes = 0;
    std::size_t driver_allocations = 0;
    std::size_t cache_hits = 0;
    std::size_t slab_suballocations = 0;
    std::size_t slab_bytes = 0;
    std::size_t slab_free_bytes = 0;
    std::size_t persistent_live_bytes = 0;
    std::size_t persistent_allocations = 0;
};

class Runtime final {
public:
    explicit Runtime(RuntimeOptions options = {});
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) noexcept = default;

    [[nodiscard]] cudaStream_t stream() const noexcept;
    [[nodiscard]] cublasLtHandle_t cublas_lt() const noexcept;
    [[nodiscard]] cudnnHandle_t cudnn() const noexcept;
    [[nodiscard]] curandGenerator_t curand() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const cudaDeviceProp& device_properties() const noexcept;
    [[nodiscard]] const RuntimeOptions& options() const noexcept;
    [[nodiscard]] FP8ExecutionMode fp8_execution_mode() const noexcept;
    [[nodiscard]] std::shared_ptr<RuntimeState> state() const noexcept { return state_; }

    void synchronize() const;
    void set_seed(std::uint64_t seed) const;
    [[nodiscard]] MemoryArenaStats memory_arena_stats() const noexcept;
    void trim_memory_arena() const;

    // Used by reusable CUDA Graph construction. During capture, temporary
    // tensors become stream-captured cudaMallocAsync/cudaFreeAsync graph nodes
    // instead of entering the normal reusable arena.
    void begin_graph_allocation_capture() const;
    void end_graph_allocation_capture() const noexcept;

private:
    std::shared_ptr<RuntimeState> state_;
};

} // namespace sdxl::cuda
