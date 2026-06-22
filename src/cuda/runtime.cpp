#include "sdxl/cuda/runtime.hpp"
#include "runtime_internal.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace sdxl::cuda {
namespace {

[[nodiscard]] std::string failure_message(std::string_view family,
                                          std::string_view expression,
                                          std::string_view file,
                                          int line,
                                          std::string_view detail) {
    std::ostringstream stream;
    stream << family << " failure at " << file << ':' << line << " while evaluating "
           << expression << ": " << detail;
    return stream.str();
}

[[nodiscard]] const char* cublas_status_name(cublasStatus_t status) noexcept {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

[[nodiscard]] const char* curand_status_name(curandStatus_t status) noexcept {
    switch (status) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
    default: return "CURAND_STATUS_UNKNOWN";
    }
}

} // namespace


AttentionBackend parse_attention_backend(std::string_view value) {
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "auto") return AttentionBackend::Auto;
    if (lowered == "cudnn" || lowered == "cudnn-sdpa" || lowered == "sdpa") {
        return AttentionBackend::CuDnnSDPA;
    }
    if (lowered == "flash" || lowered == "flash-sm80" || lowered == "flash2") {
        return AttentionBackend::FlashSM80;
    }
    if (lowered == "warp" || lowered == "warp-online" || lowered == "reference") {
        return AttentionBackend::WarpOnline;
    }
    throw CudaError("attention backend must be auto, cudnn-sdpa, flash-sm80, or warp-online");
}

const char* attention_backend_name(AttentionBackend backend) noexcept {
    switch (backend) {
    case AttentionBackend::Auto: return "auto";
    case AttentionBackend::CuDnnSDPA: return "cudnn-sdpa";
    case AttentionBackend::FlashSM80: return "flash-sm80";
    case AttentionBackend::WarpOnline: return "warp-online";
    }
    return "unknown";
}

void check_cuda(cudaError_t status, std::string_view expression, std::string_view file, int line) {
    if (status == cudaSuccess) return;
    throw CudaError(failure_message("CUDA", expression, file, line, cudaGetErrorString(status)));
}

void check_cublas(cublasStatus_t status, std::string_view expression, std::string_view file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw CudaError(failure_message("cuBLASLt", expression, file, line, cublas_status_name(status)));
}

void check_cudnn(cudnnStatus_t status, std::string_view expression, std::string_view file, int line) {
    if (status == CUDNN_STATUS_SUCCESS) return;
    throw CudaError(failure_message("cuDNN", expression, file, line, cudnnGetErrorString(status)));
}

void check_curand(curandStatus_t status, std::string_view expression, std::string_view file, int line) {
    if (status == CURAND_STATUS_SUCCESS) return;
    throw CudaError(failure_message("cuRAND", expression, file, line, curand_status_name(status)));
}


MatmulDescriptors::~MatmulDescriptors() {
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (d != nullptr) cublasLtMatrixLayoutDestroy(d);
    if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
}

ConvolutionDescriptors::~ConvolutionDescriptors() {
    if (convolution != nullptr) cudnnDestroyConvolutionDescriptor(convolution);
    if (filter != nullptr) cudnnDestroyFilterDescriptor(filter);
    if (bias != nullptr) cudnnDestroyTensorDescriptor(bias);
    if (output != nullptr) cudnnDestroyTensorDescriptor(output);
    if (input != nullptr) cudnnDestroyTensorDescriptor(input);
}

namespace {
constexpr std::size_t arena_alignment = 256;

[[nodiscard]] std::size_t align_arena(std::size_t bytes) noexcept {
    return (bytes + arena_alignment - 1) & ~(arena_alignment - 1);
}

void insert_and_coalesce(std::map<std::size_t, std::size_t>& ranges,
                         std::size_t offset,
                         std::size_t bytes) {
    auto next = ranges.lower_bound(offset);
    if (next != ranges.begin()) {
        auto previous = std::prev(next);
        if (previous->first + previous->second == offset) {
            offset = previous->first;
            bytes += previous->second;
            ranges.erase(previous);
        }
    }
    next = ranges.lower_bound(offset);
    if (next != ranges.end() && offset + bytes == next->first) {
        bytes += next->second;
        ranges.erase(next);
    }
    ranges.emplace(offset, bytes);
}
} // namespace

void* RuntimeState::acquire_block(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    bytes = align_arena(bytes);
    std::lock_guard lock(arena_mutex);
    if (graph_allocation_capture_active) {
        void* pointer = nullptr;
        SDXL_CUDA_CHECK(cudaMallocAsync(&pointer, bytes, stream));
        arena_live_bytes += bytes;
        arena_peak_live_bytes = std::max(arena_peak_live_bytes, arena_live_bytes);
        ++arena_driver_allocations;
        return pointer;
    }

    if (arena_slab != nullptr) {
        auto selected = arena_slab_free_ranges.end();
        for (auto iterator = arena_slab_free_ranges.begin();
             iterator != arena_slab_free_ranges.end(); ++iterator) {
            if (iterator->second < bytes) continue;
            if (selected == arena_slab_free_ranges.end() ||
                iterator->second < selected->second) {
                selected = iterator;
            }
        }
        if (selected != arena_slab_free_ranges.end()) {
            const std::size_t offset = selected->first;
            const std::size_t available = selected->second;
            arena_slab_free_ranges.erase(selected);
            if (available > bytes) {
                arena_slab_free_ranges.emplace(offset + bytes, available - bytes);
            }
            auto* pointer = static_cast<std::byte*>(arena_slab) + offset;
            arena_slab_live_ranges.emplace(pointer, std::pair{offset, bytes});
            arena_live_bytes += bytes;
            arena_peak_live_bytes = std::max(arena_peak_live_bytes, arena_live_bytes);
            ++arena_slab_suballocations;
            return pointer;
        }
    }

    if (options.arena_cache_limit_bytes != 0) {
        auto candidate = arena_free_blocks.find(bytes);
        if (candidate != arena_free_blocks.end()) {
            const std::size_t allocation_bytes = candidate->first;
            void* pointer = candidate->second;
            arena_free_blocks.erase(candidate);
            arena_cached_bytes -= allocation_bytes;
            arena_live_bytes += allocation_bytes;
            arena_peak_live_bytes = std::max(arena_peak_live_bytes, arena_live_bytes);
            ++arena_cache_hits;
            return pointer;
        }
    }
    void* pointer = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(&pointer, bytes, stream));
    arena_live_bytes += bytes;
    arena_peak_live_bytes = std::max(arena_peak_live_bytes, arena_live_bytes);
    ++arena_driver_allocations;
    return pointer;
}


void* RuntimeState::acquire_persistent_block(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    const std::size_t aligned = align_arena(bytes);
    void* pointer = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(&pointer, aligned, stream));
    {
        std::lock_guard lock(arena_mutex);
        persistent_live_bytes += aligned;
        ++persistent_allocations;
    }
    return pointer;
}

void RuntimeState::release_persistent_block(void* pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr) return;
    {
        std::lock_guard lock(arena_mutex);
        const std::size_t aligned = align_arena(bytes);
        persistent_live_bytes = persistent_live_bytes >= aligned
            ? persistent_live_bytes - aligned : 0;
    }
    cudaFreeAsync(pointer, stream);
}

void RuntimeState::release_block(void* pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr || bytes == 0) return;
    bytes = align_arena(bytes);
    std::lock_guard lock(arena_mutex);
    const auto slab = arena_slab_live_ranges.find(pointer);
    if (slab != arena_slab_live_ranges.end()) {
        const auto [offset, allocation_bytes] = slab->second;
        arena_live_bytes = arena_live_bytes >= allocation_bytes
            ? arena_live_bytes - allocation_bytes : 0;
        arena_slab_live_ranges.erase(slab);
        insert_and_coalesce(arena_slab_free_ranges, offset, allocation_bytes);
        return;
    }

    arena_live_bytes = arena_live_bytes >= bytes ? arena_live_bytes - bytes : 0;
    if (graph_allocation_capture_active) {
        cudaFreeAsync(pointer, stream);
        return;
    }
    const bool cache = options.arena_cache_limit_bytes != 0 &&
        bytes <= options.arena_cache_limit_bytes &&
        arena_cached_bytes <= options.arena_cache_limit_bytes - bytes;
    if (cache) {
        arena_free_blocks.emplace(bytes, pointer);
        arena_cached_bytes += bytes;
    } else {
        cudaFreeAsync(pointer, stream);
    }
}

void RuntimeState::trim_arena() noexcept {
    std::lock_guard lock(arena_mutex);
    for (const auto& [bytes, pointer] : arena_free_blocks) {
        (void)bytes;
        cudaFreeAsync(pointer, stream);
    }
    arena_free_blocks.clear();
    arena_cached_bytes = 0;
    // The persistent slab is intentionally retained. All released slab ranges
    // are already available for immediate reuse without a driver call.
}

MemoryArenaStats RuntimeState::arena_stats() const noexcept {
    std::lock_guard lock(arena_mutex);
    std::size_t slab_free = 0;
    for (const auto& [offset, bytes] : arena_slab_free_ranges) {
        (void)offset;
        slab_free += bytes;
    }
    return MemoryArenaStats{arena_live_bytes, arena_cached_bytes, arena_peak_live_bytes,
                            arena_driver_allocations, arena_cache_hits,
                            arena_slab_suballocations, arena_slab_bytes, slab_free,
                            persistent_live_bytes, persistent_allocations};
}

void RuntimeState::begin_graph_allocation_capture() {
    std::lock_guard lock(arena_mutex);
    if (graph_allocation_capture_active) {
        throw CudaError("CUDA Graph allocation capture is already active");
    }
    graph_allocation_capture_active = true;
}

void RuntimeState::end_graph_allocation_capture() noexcept {
    std::lock_guard lock(arena_mutex);
    graph_allocation_capture_active = false;
}

void* RuntimeState::ensure_cudnn_workspace(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > options.cudnn_workspace_limit_bytes) {
        throw CudaError("requested cuDNN workspace exceeds the configured limit");
    }
    if (cudnn_workspace_bytes >= bytes) return cudnn_workspace;
    void* replacement = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(&replacement, bytes, stream));
    if (cudnn_workspace != nullptr) SDXL_CUDA_CHECK(cudaFreeAsync(cudnn_workspace, stream));
    cudnn_workspace = replacement;
    cudnn_workspace_bytes = bytes;
    return cudnn_workspace;
}

RuntimeState::~RuntimeState() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    trim_arena();
    if (stream != nullptr) cudaStreamSynchronize(stream);
    convolution_plans.clear();
    matmul_plans.clear();
    if (cudnn_workspace != nullptr) cudaFree(cudnn_workspace);
    if (cublas_workspace != nullptr) cudaFree(cublas_workspace);
    if (arena_slab != nullptr) cudaFree(arena_slab);
    if (curand != nullptr) curandDestroyGenerator(curand);
    if (cudnn != nullptr) cudnnDestroy(cudnn);
    if (cublas_lt != nullptr) cublasLtDestroy(cublas_lt);
    if (stream != nullptr) cudaStreamDestroy(stream);
}

Runtime::Runtime(RuntimeOptions options) : state_(std::make_shared<RuntimeState>()) {
    state_->options = options;
    state_->device = options.device;
    SDXL_CUDA_CHECK(cudaSetDevice(options.device));
    SDXL_CUDA_CHECK(cudaGetDeviceProperties(&state_->properties, options.device));
    if (state_->properties.major < 8) {
        throw CudaError("The CUDA backend requires an Ampere-or-newer GPU (compute capability 8.0+)");
    }
    SDXL_CUDA_CHECK(cudaStreamCreateWithFlags(&state_->stream, cudaStreamNonBlocking));
    SDXL_CUBLAS_CHECK(cublasLtCreate(&state_->cublas_lt));
    SDXL_CUDNN_CHECK(cudnnCreate(&state_->cudnn));
    SDXL_CUDNN_CHECK(cudnnSetStream(state_->cudnn, state_->stream));
    SDXL_CURAND_CHECK(curandCreateGenerator(&state_->curand, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    SDXL_CURAND_CHECK(curandSetStream(state_->curand, state_->stream));

    SDXL_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&state_->memory_pool, options.device));
    std::uint64_t threshold = options.release_threshold_zero ? 0ULL : UINT64_MAX;
    SDXL_CUDA_CHECK(cudaMemPoolSetAttribute(state_->memory_pool,
                                           cudaMemPoolAttrReleaseThreshold,
                                           &threshold));

    if (options.arena_reserve_bytes != 0) {
        state_->arena_slab_bytes = align_arena(options.arena_reserve_bytes);
        SDXL_CUDA_CHECK(cudaMalloc(&state_->arena_slab, state_->arena_slab_bytes));
        state_->arena_slab_free_ranges.emplace(0, state_->arena_slab_bytes);
        ++state_->arena_driver_allocations;
    }

    if (options.cublas_workspace_bytes != 0) {
        SDXL_CUDA_CHECK(cudaMalloc(&state_->cublas_workspace, options.cublas_workspace_bytes));
    }
}

Runtime::~Runtime() = default;

cudaStream_t Runtime::stream() const noexcept { return state_->stream; }
cublasLtHandle_t Runtime::cublas_lt() const noexcept { return state_->cublas_lt; }
cudnnHandle_t Runtime::cudnn() const noexcept { return state_->cudnn; }
curandGenerator_t Runtime::curand() const noexcept { return state_->curand; }
int Runtime::device() const noexcept { return state_->device; }
const cudaDeviceProp& Runtime::device_properties() const noexcept { return state_->properties; }
const RuntimeOptions& Runtime::options() const noexcept { return state_->options; }

FP8ExecutionMode Runtime::fp8_execution_mode() const noexcept {
    const int capability = state_->properties.major * 10 + state_->properties.minor;
    const bool native_supported = capability >= 89;
    const bool weight_only_supported = capability >= 70;
    switch (state_->options.fp8_backend) {
    case FP8BackendPreference::NativeOnly:
        return native_supported ? FP8ExecutionMode::NativeTensorCore
                                : FP8ExecutionMode::Unsupported;
    case FP8BackendPreference::WeightOnly:
        return weight_only_supported ? FP8ExecutionMode::WeightOnlyTensorCore
                                     : FP8ExecutionMode::Unsupported;
    case FP8BackendPreference::Auto:
        if (native_supported) return FP8ExecutionMode::NativeTensorCore;
        if (weight_only_supported) return FP8ExecutionMode::WeightOnlyTensorCore;
        return FP8ExecutionMode::Unsupported;
    }
    return FP8ExecutionMode::Unsupported;
}

void Runtime::synchronize() const { SDXL_CUDA_CHECK(cudaStreamSynchronize(state_->stream)); }

MemoryArenaStats Runtime::memory_arena_stats() const noexcept { return state_->arena_stats(); }

void Runtime::trim_memory_arena() const {
    state_->trim_arena();
    SDXL_CUDA_CHECK(cudaStreamSynchronize(state_->stream));
}

void Runtime::begin_graph_allocation_capture() const {
    state_->begin_graph_allocation_capture();
}

void Runtime::end_graph_allocation_capture() const noexcept {
    state_->end_graph_allocation_capture();
}

void Runtime::set_seed(std::uint64_t seed) const {
    std::lock_guard lock(state_->curand_mutex);
    SDXL_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(state_->curand, seed));
    SDXL_CURAND_CHECK(curandSetGeneratorOffset(state_->curand, 0ULL));
}

} // namespace sdxl::cuda
