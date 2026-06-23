#include "sdxl/cuda/weights.hpp"
#include "int8_convrot.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace sdxl::cuda {
namespace {

enum class SourceType : int {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
    F8E4M3FN = 4,
    F8E5M2 = 5
};

struct PackMetadata {
    int rank = 0;
    std::uint64_t dimensions[4]{};
    std::int64_t strides[4]{};
    SourceType type = SourceType::F16;
};

__device__ float scale_by_power_of_two(float value, int exponent) {
    while (exponent > 0) { value *= 2.0F; --exponent; }
    while (exponent < 0) { value *= 0.5F; ++exponent; }
    return value;
}

__device__ float decode_fp8(unsigned char bits,
                            int exponent_bits,
                            int mantissa_bits,
                            int exponent_bias,
                            bool finite_only) {
    const bool negative = (bits & 0x80U) != 0;
    const unsigned exponent_mask = (1U << exponent_bits) - 1U;
    const unsigned mantissa_mask = (1U << mantissa_bits) - 1U;
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    const unsigned mantissa = bits & mantissa_mask;
    float value = 0.0F;
    if (exponent == 0) {
        if (mantissa != 0) {
            value = scale_by_power_of_two(
                static_cast<float>(mantissa) / static_cast<float>(1U << mantissa_bits),
                1 - exponent_bias);
        }
    } else if (exponent == exponent_mask && !finite_only) {
        value = mantissa == 0 ? CUDART_INF_F : CUDART_NAN_F;
    } else if (finite_only && exponent == exponent_mask && mantissa == mantissa_mask) {
        value = CUDART_NAN_F;
    } else {
        value = scale_by_power_of_two(
            1.0F + static_cast<float>(mantissa) / static_cast<float>(1U << mantissa_bits),
            static_cast<int>(exponent) - exponent_bias);
    }
    return negative ? -value : value;
}

__device__ float read_source(const unsigned char* address, SourceType type) {
    switch (type) {
    case SourceType::F16:
        return __half2float(*reinterpret_cast<const __half*>(address));
    case SourceType::BF16:
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(address));
    case SourceType::F32:
        return *reinterpret_cast<const float*>(address);
    case SourceType::F64:
        return static_cast<float>(*reinterpret_cast<const double*>(address));
    case SourceType::F8E4M3FN:
        return decode_fp8(*address, 4, 3, 7, true);
    case SourceType::F8E5M2:
        return decode_fp8(*address, 5, 2, 15, false);
    }
    return CUDART_NAN_F;
}

__device__ std::int64_t source_offset(std::size_t linear, const PackMetadata& metadata) {
    std::size_t remainder = linear;
    std::int64_t byte_offset = 0;
    for (int dimension = metadata.rank - 1; dimension >= 0; --dimension) {
        const std::uint64_t coordinate = remainder % metadata.dimensions[dimension];
        remainder /= metadata.dimensions[dimension];
        byte_offset += static_cast<std::int64_t>(coordinate) * metadata.strides[dimension];
    }
    return byte_offset;
}

__global__ void pack_to_f16_kernel(const unsigned char* source,
                                   __half* destination,
                                   std::size_t count,
                                   PackMetadata metadata) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= count) return;
    destination[linear] = __float2half_rn(read_source(source + source_offset(linear, metadata), metadata.type));
}

__global__ void pack_to_f32_kernel(const unsigned char* source,
                                   float* destination,
                                   std::size_t count,
                                   PackMetadata metadata) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= count) return;
    destination[linear] = read_source(source + source_offset(linear, metadata), metadata.type);
}

__global__ void source_amax_kernel(const unsigned char* source,
                                   std::size_t count,
                                   std::size_t row_width,
                                   unsigned int* maximum_bits,
                                   std::size_t scale_count,
                                   PackMetadata metadata) {
    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count;
         linear += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const float value = fabsf(read_source(source + source_offset(linear, metadata), metadata.type));
        if (!isfinite(value)) continue;
        const std::size_t scale_index = scale_count == 1 ? 0 : linear / row_width;
        atomicMax(maximum_bits + scale_index, __float_as_uint(value));
    }
}

__global__ void finalize_fp8_scales_kernel(float* scales,
                                           std::size_t count,
                                           float maximum_fp8) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float maximum = __uint_as_float(reinterpret_cast<unsigned int*>(scales)[index]);
    scales[index] = maximum > 0.0F ? fmaxf(maximum / maximum_fp8, 1.0e-12F) : 1.0F;
}

template <typename FP8>
__global__ void pack_to_fp8_kernel(const unsigned char* source,
                                   FP8* destination,
                                   std::size_t count,
                                   const float* scales,
                                   std::size_t scale_count,
                                   bool k_major,
                                   std::size_t rows,
                                   std::size_t columns,
                                   PackMetadata metadata) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= count) return;
    // The logical source is always [output_channel, input_feature], even
    // when one tensor-wide scale is used. Scale grouping and matrix indexing
    // must therefore remain separate: using the reduction row width here
    // would collapse every tensor-wide matrix into output row zero.
    const std::size_t output_channel = columns == 0 ? 0 : linear / columns;
    const std::size_t scale_index = scale_count == 1 ? 0 : output_channel;
    const float scale = scales[scale_index];
    const float value = read_source(source + source_offset(linear, metadata), metadata.type);
    std::size_t destination_index = linear;
    if (k_major) {
        const std::size_t feature = linear % columns;
        destination_index = feature * rows + output_channel;
    }
    destination[destination_index] = FP8(value / scale);
}

__global__ void fill_scale_kernel(float* scales, std::size_t count, float value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) scales[index] = value;
}

[[nodiscard]] bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

[[nodiscard]] bool begins_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool is_linear_weight(std::string_view logical_name, const TensorView& source) {
    if (!ends_with(logical_name, ".weight") || source.shape.size() != 2) return false;
    if (logical_name.find("embeddings.") != std::string_view::npos ||
        logical_name.find("token_embedding") != std::string_view::npos ||
        logical_name.find("position_embedding") != std::string_view::npos) {
        return false;
    }
    return true;
}

[[nodiscard]] SourceType source_type(DType type) {
    switch (type) {
    case DType::F16: return SourceType::F16;
    case DType::BF16: return SourceType::BF16;
    case DType::F32: return SourceType::F32;
    case DType::F64: return SourceType::F64;
    case DType::F8E4M3FN: return SourceType::F8E4M3FN;
    case DType::F8E5M2: return SourceType::F8E5M2;
    default:
        throw CudaError("CUDA weights require floating-point tensors; received " +
                        std::string(dtype_name(type)));
    }
}

[[nodiscard]] std::size_t span_bytes(const TensorView& source) {
    if (source.shape.empty() || source.shape.size() != source.strides_bytes.size()) {
        throw CudaError("invalid mapped weight view: " + source.source_key);
    }
    std::uint64_t maximum = 0;
    for (std::size_t dimension = 0; dimension < source.shape.size(); ++dimension) {
        if (source.strides_bytes[dimension] < 0) {
            throw CudaError("negative weight strides are not supported: " + source.source_key);
        }
        if (source.shape[dimension] != 0) {
            maximum += (source.shape[dimension] - 1) *
                       static_cast<std::uint64_t>(source.strides_bytes[dimension]);
        }
    }
    maximum += dtype_size(source.dtype);
    if (maximum > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw CudaError("weight span exceeds host addressable range: " + source.source_key);
    }
    return static_cast<std::size_t>(maximum);
}

[[nodiscard]] std::size_t resident_bytes(const Tensor& tensor) {
    return tensor.bytes() + tensor.dequant_scale_count() * sizeof(float);
}

[[nodiscard]] ScalarType fp8_type_for(const TensorView& source,
                                      const FP8WeightLoadOptions& options) {
    if (options.format == FP8Format::E4M3) return ScalarType::Float8E4M3;
    if (options.format == FP8Format::E5M2) return ScalarType::Float8E5M2;
    if (options.preserve_native_fp8 && source.dtype == DType::F8E5M2) {
        return ScalarType::Float8E5M2;
    }
    return ScalarType::Float8E4M3;
}

void update_stats(WeightLoadStats& stats, const Tensor& tensor) {
    ++stats.tensors;
    switch (tensor.type()) {
    case ScalarType::Float8E4M3: ++stats.fp8_e4m3_tensors; break;
    case ScalarType::Float8E5M2: ++stats.fp8_e5m2_tensors; break;
    case ScalarType::Int8:
        ++stats.int8_tensors;
        if (tensor.uses_convrot()) ++stats.int8_convrot_tensors;
        break;
    case ScalarType::Float16: ++stats.fp16_tensors; break;
    case ScalarType::Float32: ++stats.fp32_tensors; break;
    case ScalarType::Int32: break;
    }
    if (tensor.dequant_scale_mode() == FP8ScaleMode::TensorWide) ++stats.tensorwide_scaled_tensors;
    if (tensor.dequant_scale_mode() == FP8ScaleMode::PerOutputChannel) ++stats.channel_scaled_tensors;
}


template <typename T>
void write_binary(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) throw CudaError("failed while writing FP8 cache");
}

template <typename T>
[[nodiscard]] T read_binary(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw CudaError("truncated FP8 cache");
    return value;
}

void write_string(std::ostream& output, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw CudaError("FP8 cache string is too long");
    }
    write_binary(output, static_cast<std::uint32_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) throw CudaError("failed while writing FP8 cache string");
}

[[nodiscard]] std::string read_string(std::istream& input) {
    const std::uint32_t size = read_binary<std::uint32_t>(input);
    if (size > 16U * 1024U * 1024U) throw CudaError("invalid FP8 cache string size");
    std::string value(size, '\0');
    input.read(value.data(), static_cast<std::streamsize>(value.size()));
    if (!input) throw CudaError("truncated FP8 cache string");
    return value;
}

constexpr std::array<char, 8> fp8_cache_magic{{'S','D','X','L','F','8','C','3'}};

} // namespace

WeightStore::WeightStore(const Runtime& runtime, const SDXLModel& model)
    : runtime_(&runtime), model_(&model) {}

Tensor WeightStore::upload_tensor(const TensorView& source,
                                  ScalarType destination_type,
                                  TensorRole role,
                                  FP8ScaleMode scale_mode) const {
    if (runtime_ == nullptr) throw CudaError("CUDA weight store has no runtime");
    if (source.shape.size() > 4) {
        throw CudaError("CUDA weight packing supports rank <= 4: " + source.source_key);
    }
    std::vector<std::size_t> shape;
    shape.reserve(source.shape.size());
    for (const std::uint64_t dimension : source.shape) {
        shape.push_back(static_cast<std::size_t>(dimension));
    }
    if (!is_fp8(destination_type) && destination_type != ScalarType::Float16 &&
        destination_type != ScalarType::Float32) {
        throw CudaError("CUDA weight upload target must be FP8, float16, or float32");
    }

    const bool same_native_fp8 =
        (destination_type == ScalarType::Float8E4M3 && source.dtype == DType::F8E4M3FN) ||
        (destination_type == ScalarType::Float8E5M2 && source.dtype == DType::F8E5M2);
    const std::size_t scale_count = scale_mode == FP8ScaleMode::PerOutputChannel
        ? (shape.size() == 2 ? shape[0] : 0) : 1;
    if (is_fp8(destination_type) && scale_count == 0) {
        throw CudaError("per-output-channel FP8 scaling requires a rank-2 matrix");
    }
    const bool k_major_fp8 = is_fp8(destination_type) && shape.size() == 2 &&
        runtime_->fp8_execution_mode() == FP8ExecutionMode::WeightOnlyTensorCore;

    if (is_fp8(destination_type) && same_native_fp8 && source.contiguous() && !k_major_fp8) {
        Tensor destination = Tensor::allocate_persistent(
            *runtime_, shape, destination_type, role);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data, destination.bytes(),
                                        cudaMemcpyHostToDevice, runtime_->stream()));
        Tensor scales = Tensor::allocate_persistent(
            *runtime_, {scale_count}, ScalarType::Float32,
            TensorRole::FP8ScaleMetadata);
        constexpr unsigned threads = 256;
        fill_scale_kernel<<<static_cast<unsigned>((scale_count + threads - 1) / threads),
                            threads, 0, runtime_->stream()>>>(
            scales.float_data(), scale_count, 1.0F);
        SDXL_CUDA_CHECK(cudaGetLastError());
        destination.attach_dequant_scale(scales, scale_mode);
        return destination;
    }

    Tensor destination = Tensor::allocate_persistent(
            *runtime_, shape, destination_type, role);
    if (k_major_fp8) destination.set_fp8_storage_layout(FP8StorageLayout::KMajorKN);
    if (((source.dtype == DType::F16 && destination_type == ScalarType::Float16) ||
         (source.dtype == DType::F32 && destination_type == ScalarType::Float32)) &&
        source.contiguous()) {
        SDXL_CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data, destination.bytes(),
                                        cudaMemcpyHostToDevice, runtime_->stream()));
        return destination;
    }

    const std::size_t source_span = span_bytes(source);
    unsigned char* staging = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&staging), source_span,
                                    runtime_->stream()));
    SDXL_CUDA_CHECK(cudaMemcpyAsync(staging, source.data, source_span,
                                    cudaMemcpyHostToDevice, runtime_->stream()));

    PackMetadata metadata;
    metadata.rank = static_cast<int>(source.shape.size());
    metadata.type = source_type(source.dtype);
    for (std::size_t dimension = 0; dimension < source.shape.size(); ++dimension) {
        metadata.dimensions[dimension] = source.shape[dimension];
        metadata.strides[dimension] = source.strides_bytes[dimension];
    }
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((destination.elements() + threads - 1) / threads);
    if (destination_type == ScalarType::Float16) {
        pack_to_f16_kernel<<<blocks, threads, 0, runtime_->stream()>>>(
            staging, destination.half_data(), destination.elements(), metadata);
    } else if (destination_type == ScalarType::Float32) {
        pack_to_f32_kernel<<<blocks, threads, 0, runtime_->stream()>>>(
            staging, destination.float_data(), destination.elements(), metadata);
    } else {
        // Quantize directly from the checkpoint's mapped dtype. No full FP16 or FP32
        // expansion of the matrix is created on the device.
        Tensor scales = Tensor::zeros_persistent(
            *runtime_, {scale_count}, ScalarType::Float32,
            TensorRole::FP8ScaleMetadata);
        const std::size_t row_width = scale_count == 1 ? destination.elements() : shape[1];
        const unsigned reduction_blocks = static_cast<unsigned>(std::max<std::size_t>(
            1, std::min<std::size_t>(4096, (destination.elements() + threads - 1) / threads)));
        source_amax_kernel<<<reduction_blocks, threads, 0, runtime_->stream()>>>(
            staging, destination.elements(), row_width,
            reinterpret_cast<unsigned int*>(scales.float_data()), scale_count, metadata);
        SDXL_CUDA_CHECK(cudaGetLastError());
        const float maximum = destination_type == ScalarType::Float8E4M3 ? 448.0F : 57344.0F;
        finalize_fp8_scales_kernel<<<static_cast<unsigned>((scale_count + threads - 1) / threads),
                                     threads, 0, runtime_->stream()>>>(
            scales.float_data(), scale_count, maximum);
        SDXL_CUDA_CHECK(cudaGetLastError());
        if (destination_type == ScalarType::Float8E4M3) {
            pack_to_fp8_kernel<<<blocks, threads, 0, runtime_->stream()>>>(
                staging, reinterpret_cast<__nv_fp8_e4m3*>(destination.data()),
                destination.elements(), scales.float_data(), scale_count,
                k_major_fp8, shape[0], shape[1], metadata);
        } else {
            pack_to_fp8_kernel<<<blocks, threads, 0, runtime_->stream()>>>(
                staging, reinterpret_cast<__nv_fp8_e5m2*>(destination.data()),
                destination.elements(), scales.float_data(), scale_count,
                k_major_fp8, shape[0], shape[1], metadata);
        }
        SDXL_CUDA_CHECK(cudaGetLastError());
        destination.attach_dequant_scale(scales, scale_mode);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    SDXL_CUDA_CHECK(cudaFreeAsync(staging, runtime_->stream()));
    return destination;
}

WeightLoadStats WeightStore::load_prefix(std::string_view prefix, ScalarType destination_type) {
    return load_prefixes({std::string(prefix)}, destination_type);
}

WeightLoadStats WeightStore::load_prefixes(const std::vector<std::string>& prefixes,
                                            ScalarType destination_type) {
    if (model_ == nullptr || runtime_ == nullptr) throw CudaError("CUDA weight store is not initialized");
    WeightLoadStats stats;
    for (const auto& [name, slot] : model_->graph().parameter_index()) {
        bool selected = false;
        for (const std::string& prefix : prefixes) {
            if (begins_with(name, prefix)) {
                selected = true;
                break;
            }
        }
        if (!selected) continue;
        const TensorRole role = begins_with(name, "vae.") ? TensorRole::VAE : TensorRole::Model;
        const auto existing = tensors_.find(name);
        if (existing != tensors_.end()) {
            if (existing->second.type() == destination_type && existing->second.role() == role) continue;
            device_bytes_ -= resident_bytes(existing->second);
            tensors_.erase(existing);
        }
        if (slot == nullptr || !slot->tensor.has_value()) {
            throw CudaError("required model weight is not loaded on the host: " + name);
        }
        Tensor uploaded = upload_tensor(*slot->tensor, destination_type, role);
        stats.source_bytes += static_cast<std::size_t>(slot->tensor->logical_bytes());
        stats.device_bytes += resident_bytes(uploaded);
        update_stats(stats, uploaded);
        device_bytes_ += resident_bytes(uploaded);
        tensors_.emplace(name, std::move(uploaded));
    }
    runtime_->synchronize();
    return stats;
}

WeightLoadStats WeightStore::load_unet_fp8(FP8WeightLoadOptions options,
                                                  const FP8CacheOptions* cache,
                                                  FP8CacheStats* cache_stats) {
    if (model_ == nullptr || runtime_ == nullptr) throw CudaError("CUDA weight store is not initialized");
    if (runtime_->fp8_execution_mode() == FP8ExecutionMode::Unsupported) {
        throw CudaError("selected FP8 backend is unsupported on this GPU");
    }
    if (options.backend == FP8Backend::Native &&
        runtime_->fp8_execution_mode() != FP8ExecutionMode::NativeTensorCore) {
        throw CudaError("native FP8 was requested but the runtime is not using native FP8 Tensor Cores");
    }
    active_unet_fp8_options_ = options;
    WeightLoadStats stats;
    FP8CacheStats local_cache_stats;
    FP8CacheStats& cache_result = cache_stats != nullptr ? *cache_stats : local_cache_stats;
    if (cache != nullptr && cache->read && !cache->path.empty() &&
        std::filesystem::is_regular_file(cache->path)) {
        try {
            cache_result.hit = load_fp8_cache(*cache, stats, cache_result);
        } catch (const std::exception&) {
            // A stale, partial, or incompatible cache is ignored and rebuilt.
            cache_result = {};
        }
    }
    for (const auto& [name, slot] : model_->graph().parameter_index()) {
        if (!begins_with(name, "unet.")) continue;
        if (slot == nullptr || !slot->tensor.has_value()) {
            throw CudaError("required UNet weight is not loaded on the host: " + name);
        }
        const bool fp8_eligible = options.quantize_linear_weights &&
                                  slot->expected_shape.size() == 2 &&
                                  ends_with(name, ".weight");
        const ScalarType destination_type = fp8_eligible
            ? fp8_type_for(*slot->tensor, options) : ScalarType::Float16;
        const FP8ScaleMode scale_mode = fp8_eligible ? options.scale_mode : FP8ScaleMode::TensorWide;
        const auto existing = tensors_.find(name);
        if (existing != tensors_.end()) {
            const bool same = existing->second.type() == destination_type &&
                              (!fp8_eligible || existing->second.dequant_scale_mode() == scale_mode);
            if (same) continue;
            device_bytes_ -= resident_bytes(existing->second);
            tensors_.erase(existing);
        }
        Tensor uploaded = upload_tensor(*slot->tensor, destination_type,
                                        TensorRole::Model, scale_mode);
        stats.source_bytes += static_cast<std::size_t>(slot->tensor->logical_bytes());
        stats.device_bytes += resident_bytes(uploaded);
        update_stats(stats, uploaded);
        device_bytes_ += resident_bytes(uploaded);
        tensors_.emplace(name, std::move(uploaded));
    }
    runtime_->synchronize();
    if (cache != nullptr && cache->write && !cache->path.empty() && !cache_result.hit) {
        write_fp8_cache(*cache, cache_result);
    }
    return stats;
}

bool WeightStore::load_fp8_cache(const FP8CacheOptions& cache,
                                 WeightLoadStats& stats,
                                 FP8CacheStats& cache_stats) {
    std::ifstream input(cache.path, std::ios::binary);
    if (!input) return false;
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != fp8_cache_magic) return false;
    if (read_string(input) != cache.key) return false;
    const std::uint32_t entry_count = read_binary<std::uint32_t>(input);
    if (entry_count == 0 || entry_count > 100000U) return false;

    struct CachedEntry {
        std::string name;
        ScalarType type = ScalarType::Float8E4M3;
        FP8ScaleMode scale_mode = FP8ScaleMode::TensorWide;
        FP8StorageLayout storage_layout = FP8StorageLayout::RowMajorNK;
        std::vector<std::size_t> shape;
        std::vector<std::uint8_t> data;
        std::vector<float> scales;
    };
    std::vector<CachedEntry> entries;
    entries.reserve(entry_count);
    for (std::uint32_t index = 0; index < entry_count; ++index) {
        CachedEntry entry;
        entry.name = read_string(input);
        entry.type = static_cast<ScalarType>(read_binary<std::uint8_t>(input));
        entry.scale_mode = static_cast<FP8ScaleMode>(read_binary<std::uint8_t>(input));
        const std::uint8_t rank = read_binary<std::uint8_t>(input);
        entry.storage_layout = static_cast<FP8StorageLayout>(read_binary<std::uint8_t>(input));
        if (!is_fp8(entry.type) || rank == 0 || rank > 4) return false;
        if (entry.storage_layout != FP8StorageLayout::RowMajorNK &&
            entry.storage_layout != FP8StorageLayout::KMajorKN) return false;
        const FP8StorageLayout expected_layout =
            runtime_->fp8_execution_mode() == FP8ExecutionMode::WeightOnlyTensorCore
                ? FP8StorageLayout::KMajorKN : FP8StorageLayout::RowMajorNK;
        if (entry.storage_layout != expected_layout) return false;
        entry.shape.resize(rank);
        std::size_t elements = 1;
        for (std::uint8_t dimension = 0; dimension < rank; ++dimension) {
            const std::uint64_t size = read_binary<std::uint64_t>(input);
            if (size == 0 || size > std::numeric_limits<std::size_t>::max()) return false;
            entry.shape[dimension] = static_cast<std::size_t>(size);
            if (elements > std::numeric_limits<std::size_t>::max() / entry.shape[dimension]) return false;
            elements *= entry.shape[dimension];
        }
        const std::uint64_t data_bytes = read_binary<std::uint64_t>(input);
        const std::uint64_t scale_count = read_binary<std::uint64_t>(input);
        if (data_bytes != elements || scale_count == 0 ||
            data_bytes > std::numeric_limits<std::size_t>::max() ||
            scale_count > std::numeric_limits<std::size_t>::max()) return false;
        const ParameterSlot* slot = model_->graph().find_parameter(entry.name);
        if (slot == nullptr || !begins_with(entry.name, "unet.")) return false;
        std::vector<std::uint64_t> expected;
        expected.reserve(entry.shape.size());
        for (const std::size_t value : entry.shape) expected.push_back(value);
        if (slot->expected_shape != expected) return false;
        const std::size_t expected_scales = entry.scale_mode == FP8ScaleMode::PerOutputChannel
            ? entry.shape.front() : 1;
        if (scale_count != expected_scales) return false;
        entry.data.resize(static_cast<std::size_t>(data_bytes));
        entry.scales.resize(static_cast<std::size_t>(scale_count));
        input.read(reinterpret_cast<char*>(entry.data.data()),
                   static_cast<std::streamsize>(entry.data.size()));
        input.read(reinterpret_cast<char*>(entry.scales.data()),
                   static_cast<std::streamsize>(entry.scales.size() * sizeof(float)));
        if (!input) return false;
        entries.push_back(std::move(entry));
    }

    for (CachedEntry& entry : entries) {
        Tensor tensor = Tensor::allocate_persistent(
            *runtime_, entry.shape, entry.type, TensorRole::Model);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(tensor.data(), entry.data.data(), entry.data.size(),
                                        cudaMemcpyHostToDevice, runtime_->stream()));
        tensor.set_fp8_storage_layout(entry.storage_layout);
        Tensor scales = Tensor::allocate_persistent(
            *runtime_, {entry.scales.size()}, ScalarType::Float32,
            TensorRole::FP8ScaleMetadata);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(scales.data(), entry.scales.data(),
                                        entry.scales.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, runtime_->stream()));
        tensor.attach_dequant_scale(scales, entry.scale_mode);
        const auto existing = tensors_.find(entry.name);
        if (existing != tensors_.end()) {
            device_bytes_ -= resident_bytes(existing->second);
            tensors_.erase(existing);
        }
        stats.device_bytes += resident_bytes(tensor);
        update_stats(stats, tensor);
        device_bytes_ += resident_bytes(tensor);
        cache_stats.bytes_loaded += entry.data.size() + entry.scales.size() * sizeof(float);
        ++cache_stats.tensors_loaded;
        tensors_.emplace(std::move(entry.name), std::move(tensor));
    }
    runtime_->synchronize();
    return true;
}

void WeightStore::write_fp8_cache(const FP8CacheOptions& cache,
                                  FP8CacheStats& cache_stats) const {
    std::vector<std::pair<std::string, const Tensor*>> entries;
    for (const auto& [name, tensor] : tensors_) {
        if (begins_with(name, "unet.") && is_fp8(tensor.type())) {
            entries.emplace_back(name, &tensor);
        }
    }
    if (entries.empty()) return;
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    if (!cache.path.parent_path().empty()) {
        std::filesystem::create_directories(cache.path.parent_path());
    }
    const std::filesystem::path temporary = cache.path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw CudaError("cannot create FP8 cache: " + temporary.string());
    output.write(fp8_cache_magic.data(), static_cast<std::streamsize>(fp8_cache_magic.size()));
    write_string(output, cache.key);
    write_binary(output, static_cast<std::uint32_t>(entries.size()));
    runtime_->synchronize();
    for (const auto& [name, tensor_pointer] : entries) {
        const Tensor& tensor = *tensor_pointer;
        write_string(output, name);
        write_binary(output, static_cast<std::uint8_t>(tensor.type()));
        write_binary(output, static_cast<std::uint8_t>(tensor.dequant_scale_mode()));
        write_binary(output, static_cast<std::uint8_t>(tensor.rank()));
        write_binary(output, static_cast<std::uint8_t>(tensor.fp8_storage_layout()));
        for (const std::size_t dimension : tensor.shape()) {
            write_binary(output, static_cast<std::uint64_t>(dimension));
        }
        write_binary(output, static_cast<std::uint64_t>(tensor.bytes()));
        write_binary(output, static_cast<std::uint64_t>(tensor.dequant_scale_count()));
        std::vector<std::uint8_t> data(tensor.bytes());
        std::vector<float> scales(tensor.dequant_scale_count());
        SDXL_CUDA_CHECK(cudaMemcpy(data.data(), tensor.data(), data.size(), cudaMemcpyDeviceToHost));
        SDXL_CUDA_CHECK(cudaMemcpy(scales.data(), tensor.dequant_scale_data(),
                                   scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
        output.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
        output.write(reinterpret_cast<const char*>(scales.data()),
                     static_cast<std::streamsize>(scales.size() * sizeof(float)));
        cache_stats.bytes_written += data.size() + scales.size() * sizeof(float);
    }
    output.close();
    if (!output) throw CudaError("failed while finalizing FP8 cache");
    std::error_code error;
    std::filesystem::remove(cache.path, error);
    error.clear();
    std::filesystem::rename(temporary, cache.path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw CudaError("cannot install FP8 cache: " + error.message());
    }
    cache_stats.written = true;
}

WeightLoadStats WeightStore::load_prefixes_int8(
    const std::vector<std::string>& prefixes,
    INT8WeightLoadOptions options) {
    if (model_ == nullptr || runtime_ == nullptr) {
        throw CudaError("CUDA weight store is not initialized");
    }
    if (options.enable_convrot && !valid_convrot_group_size(options.convrot_group_size)) {
        throw CudaError("INT8 ConvRot group size must be a power of four between 4 and 256");
    }
    active_int8_options_ = options;
    WeightLoadStats stats;
    for (const auto& [name, slot] : model_->graph().parameter_index()) {
        bool selected = false;
        for (const std::string& prefix : prefixes) {
            if (begins_with(name, prefix)) {
                selected = true;
                break;
            }
        }
        if (!selected) continue;
        if (slot == nullptr || !slot->tensor.has_value()) {
            throw CudaError("required model weight is not loaded on the host: " + name);
        }
        if (tensors_.find(name) != tensors_.end()) continue;

        const TensorView& source = *slot->tensor;
        const TensorRole role = begins_with(name, "vae.") ? TensorRole::VAE : TensorRole::Model;
        Tensor destination;
        if (is_linear_weight(name, source)) {
            try {
                const QuantizationMetadata* metadata = slot->quantization.has_value()
                    ? &*slot->quantization : nullptr;
                destination = upload_or_quantize_int8_weight(
                    *runtime_, source, metadata, role, options);
            } catch (const CudaError&) {
                if (options.strict || source.dtype == DType::I8) throw;
                destination = upload_tensor(source, ScalarType::Float16, role);
            }
        } else {
            if (source.dtype == DType::I8) {
                throw CudaError("INT8 checkpoint contains a non-linear INT8 tensor that cannot be executed natively: " +
                                name);
            }
            destination = upload_tensor(source, ScalarType::Float16, role);
        }

        stats.source_bytes += static_cast<std::size_t>(source.logical_bytes());
        stats.device_bytes += resident_bytes(destination);
        update_stats(stats, destination);
        device_bytes_ += resident_bytes(destination);
        tensors_.emplace(name, std::move(destination));
    }
    runtime_->synchronize();
    return stats;
}

void WeightStore::unload_prefix(std::string_view prefix) {
    for (auto iterator = tensors_.begin(); iterator != tensors_.end();) {
        if (begins_with(iterator->first, prefix)) {
            device_bytes_ -= resident_bytes(iterator->second);
            iterator = tensors_.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

void WeightStore::clear() {
    tensors_.clear();
    device_bytes_ = 0;
}

bool WeightStore::contains(std::string_view logical_name) const {
    return tensors_.contains(std::string(logical_name));
}

const Tensor& WeightStore::get(std::string_view logical_name) const {
    const auto iterator = tensors_.find(std::string(logical_name));
    if (iterator == tensors_.end()) {
        throw CudaError("CUDA weight is not resident: " + std::string(logical_name));
    }
    return iterator->second;
}

} // namespace sdxl::cuda
