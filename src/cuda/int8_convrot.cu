#include "int8_convrot.hpp"
#include "runtime_internal.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace sdxl::cuda {
namespace {

enum class SourceType : int {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
    I8 = 4
};

struct PackMetadata {
    int rank = 0;
    std::uint64_t dimensions[4]{};
    std::int64_t strides[4]{};
    SourceType type = SourceType::F16;
};

[[nodiscard]] SourceType source_type(DType type) {
    switch (type) {
    case DType::F16: return SourceType::F16;
    case DType::BF16: return SourceType::BF16;
    case DType::F32: return SourceType::F32;
    case DType::F64: return SourceType::F64;
    case DType::I8: return SourceType::I8;
    default: break;
    }
    throw CudaError("native INT8 loading requires I8, F16, BF16, F32, or F64 weights");
}

[[nodiscard]] std::size_t source_span_bytes(const TensorView& source) {
    if (source.shape.empty() || source.shape.size() != source.strides_bytes.size()) {
        throw CudaError("invalid mapped INT8 source tensor: " + source.source_key);
    }
    std::uint64_t maximum = 0;
    for (std::size_t dimension = 0; dimension < source.shape.size(); ++dimension) {
        if (source.strides_bytes[dimension] < 0) {
            throw CudaError("negative INT8 source strides are not supported: " + source.source_key);
        }
        if (source.shape[dimension] != 0) {
            maximum += (source.shape[dimension] - 1) *
                       static_cast<std::uint64_t>(source.strides_bytes[dimension]);
        }
    }
    maximum += dtype_size(source.dtype);
    if (maximum > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw CudaError("INT8 source span exceeds addressable range: " + source.source_key);
    }
    return static_cast<std::size_t>(maximum);
}

[[nodiscard]] PackMetadata pack_metadata(const TensorView& source) {
    if (source.shape.size() > 4) {
        throw CudaError("native INT8 packing supports rank <= 4: " + source.source_key);
    }
    PackMetadata metadata;
    metadata.rank = static_cast<int>(source.shape.size());
    metadata.type = source_type(source.dtype);
    for (std::size_t i = 0; i < source.shape.size(); ++i) {
        metadata.dimensions[i] = source.shape[i];
        metadata.strides[i] = source.strides_bytes[i];
    }
    return metadata;
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

__device__ float read_source(const unsigned char* address, SourceType type) {
    switch (type) {
    case SourceType::F16: return __half2float(*reinterpret_cast<const __half*>(address));
    case SourceType::BF16: return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(address));
    case SourceType::F32: return *reinterpret_cast<const float*>(address);
    case SourceType::F64: return static_cast<float>(*reinterpret_cast<const double*>(address));
    case SourceType::I8: return static_cast<float>(*reinterpret_cast<const std::int8_t*>(address));
    }
    return 0.0F;
}

__global__ void pack_i8_kernel(const unsigned char* source,
                               std::int8_t* destination,
                               std::size_t count,
                               PackMetadata metadata) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = static_cast<std::int8_t>(
        read_source(source + source_offset(index, metadata), metadata.type));
}

__global__ void pack_f32_kernel(const unsigned char* source,
                                float* destination,
                                std::size_t count,
                                PackMetadata metadata) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = read_source(source + source_offset(index, metadata), metadata.type);
}

__device__ float* regular_hadamard_transform(float* primary,
                                              float* scratch,
                                              std::size_t group_size) {
    float* source = primary;
    float* destination = scratch;
    for (std::size_t stride = 1; stride < group_size; stride *= 4) {
        const std::size_t index = threadIdx.x;
        if (index < group_size) {
            const std::size_t block = index / (4 * stride);
            const std::size_t offset = index % stride;
            const std::size_t lane = (index / stride) % 4;
            const std::size_t base = block * 4 * stride + offset;
            const float a = source[base];
            const float b = source[base + stride];
            const float c = source[base + 2 * stride];
            const float d = source[base + 3 * stride];
            float value = 0.0F;
            if (lane == 0) value = a + b + c - d;
            else if (lane == 1) value = a + b - c + d;
            else if (lane == 2) value = a - b + c + d;
            else value = -a + b + c + d;
            destination[index] = value;
        }
        __syncthreads();
        float* swap = source;
        source = destination;
        destination = swap;
    }
    return source;
}

__device__ void atomic_max_positive_float(unsigned int* address, float value) {
    if (isfinite(value) && value >= 0.0F) atomicMax(address, __float_as_uint(value));
}

__global__ void weight_amax_kernel(const unsigned char* source,
                                   float* row_scales,
                                   std::size_t rows,
                                   std::size_t columns,
                                   std::size_t convrot_group_size,
                                   PackMetadata metadata) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ unsigned char shared_bytes[];
    float* primary = reinterpret_cast<float*>(shared_bytes);
    float* scratch = primary + convrot_group_size;
    __shared__ unsigned int maximum_bits;
    if (threadIdx.x == 0) maximum_bits = 0;
    __syncthreads();

    if (convrot_group_size == 0) {
        for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
            const std::size_t logical = row * columns + column;
            const float value = fabsf(read_source(
                source + source_offset(logical, metadata), metadata.type));
            atomic_max_positive_float(&maximum_bits, value);
        }
    } else {
        const float normalization = rsqrtf(static_cast<float>(convrot_group_size));
        for (std::size_t group = 0; group < columns; group += convrot_group_size) {
            if (threadIdx.x < convrot_group_size) {
                const std::size_t logical = row * columns + group + threadIdx.x;
                primary[threadIdx.x] = read_source(
                    source + source_offset(logical, metadata), metadata.type);
            }
            __syncthreads();
            float* transformed = regular_hadamard_transform(
                primary, scratch, convrot_group_size);
            if (threadIdx.x < convrot_group_size) {
                atomic_max_positive_float(
                    &maximum_bits, fabsf(transformed[threadIdx.x] * normalization));
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float maximum = __uint_as_float(maximum_bits);
        row_scales[row] = maximum > 0.0F ? fmaxf(maximum / 127.0F, 1.0e-30F) : 1.0F;
    }
}

__global__ void quantize_weight_kernel(const unsigned char* source,
                                       std::int8_t* destination,
                                       const float* row_scales,
                                       std::size_t rows,
                                       std::size_t columns,
                                       std::size_t convrot_group_size,
                                       PackMetadata metadata) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    const float inverse_scale = 1.0F / row_scales[row];
    extern __shared__ float shared[];
    float* primary = shared;
    float* scratch = primary + convrot_group_size;

    if (convrot_group_size == 0) {
        for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
            const std::size_t logical = row * columns + column;
            const float value = read_source(
                source + source_offset(logical, metadata), metadata.type);
            int quantized = __float2int_rn(value * inverse_scale);
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            destination[logical] = static_cast<std::int8_t>(quantized);
        }
        return;
    }

    const float normalization = rsqrtf(static_cast<float>(convrot_group_size));
    for (std::size_t group = 0; group < columns; group += convrot_group_size) {
        if (threadIdx.x < convrot_group_size) {
            const std::size_t logical = row * columns + group + threadIdx.x;
            primary[threadIdx.x] = read_source(
                source + source_offset(logical, metadata), metadata.type);
        }
        __syncthreads();
        float* transformed = regular_hadamard_transform(primary, scratch, convrot_group_size);
        if (threadIdx.x < convrot_group_size) {
            const float value = transformed[threadIdx.x] * normalization;
            int quantized = __float2int_rn(value * inverse_scale);
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            destination[row * columns + group + threadIdx.x] =
                static_cast<std::int8_t>(quantized);
        }
        __syncthreads();
    }
}

template <typename Input>
__device__ float input_value(const Input* data, std::size_t index);

template <>
__device__ float input_value<__half>(const __half* data, std::size_t index) {
    return __half2float(data[index]);
}

template <>
__device__ float input_value<float>(const float* data, std::size_t index) {
    return data[index];
}

template <typename Input>
__global__ void activation_amax_kernel(const Input* input,
                                       float* row_scales,
                                       std::size_t rows,
                                       std::size_t columns,
                                       std::size_t convrot_group_size) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* primary = shared;
    float* scratch = primary + convrot_group_size;
    __shared__ unsigned int maximum_bits;
    if (threadIdx.x == 0) maximum_bits = 0;
    __syncthreads();

    if (convrot_group_size == 0) {
        for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
            atomic_max_positive_float(
                &maximum_bits, fabsf(input_value(input, row * columns + column)));
        }
    } else {
        const float normalization = rsqrtf(static_cast<float>(convrot_group_size));
        for (std::size_t group = 0; group < columns; group += convrot_group_size) {
            if (threadIdx.x < convrot_group_size) {
                primary[threadIdx.x] =
                    input_value(input, row * columns + group + threadIdx.x);
            }
            __syncthreads();
            float* transformed = regular_hadamard_transform(
                primary, scratch, convrot_group_size);
            if (threadIdx.x < convrot_group_size) {
                atomic_max_positive_float(
                    &maximum_bits, fabsf(transformed[threadIdx.x] * normalization));
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float maximum = __uint_as_float(maximum_bits);
        row_scales[row] = maximum > 0.0F
            ? fmaxf(maximum / 127.0F, 1.0e-30F) : 1.0F;
    }
}

template <typename Input>
__global__ void quantize_activation_kernel(const Input* input,
                                           std::int8_t* output,
                                           const float* row_scales,
                                           std::size_t rows,
                                           std::size_t columns,
                                           std::size_t convrot_group_size) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    const float inverse_scale = 1.0F / row_scales[row];
    extern __shared__ float shared[];
    float* primary = shared;
    float* scratch = primary + convrot_group_size;

    if (convrot_group_size == 0) {
        for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
            int quantized = __float2int_rn(
                input_value(input, row * columns + column) * inverse_scale);
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            output[row * columns + column] = static_cast<std::int8_t>(quantized);
        }
        return;
    }

    const float normalization = rsqrtf(static_cast<float>(convrot_group_size));
    for (std::size_t group = 0; group < columns; group += convrot_group_size) {
        if (threadIdx.x < convrot_group_size) {
            primary[threadIdx.x] =
                input_value(input, row * columns + group + threadIdx.x);
        }
        __syncthreads();
        float* transformed = regular_hadamard_transform(
            primary, scratch, convrot_group_size);
        if (threadIdx.x < convrot_group_size) {
            int quantized = __float2int_rn(
                transformed[threadIdx.x] * normalization * inverse_scale);
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            output[row * columns + group + threadIdx.x] =
                static_cast<std::int8_t>(quantized);
        }
        __syncthreads();
    }
}

__global__ void int8_gemm_dp4a_kernel(const std::int8_t* activations,
                                      const std::int8_t* weights,
                                      std::int32_t* accumulator,
                                      std::size_t rows,
                                      std::size_t output_columns,
                                      std::size_t inner) {
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_columns) return;
    const int* activation4 = reinterpret_cast<const int*>(activations + row * inner);
    const int* weight4 = reinterpret_cast<const int*>(weights + column * inner);
    std::int32_t sum = 0;
    for (std::size_t item = 0; item < inner / 4; ++item) {
        sum = __dp4a(activation4[item], weight4[item], sum);
    }
    accumulator[row * output_columns + column] = sum;
}

__global__ void dequantize_gemm_f32_kernel(const std::int32_t* accumulator,
                                           float* output,
                                           const float* activation_scales,
                                           const float* weight_scales,
                                           std::size_t rows,
                                           std::size_t columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rows * columns;
    if (index >= count) return;
    const std::size_t row = index / columns;
    const std::size_t column = index % columns;
    output[index] = static_cast<float>(accumulator[index]) *
                    activation_scales[row] * weight_scales[column];
}

__global__ void dequantize_gemm_f16_kernel(const std::int32_t* accumulator,
                                           __half* output,
                                           const float* activation_scales,
                                           const float* weight_scales,
                                           std::size_t rows,
                                           std::size_t columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rows * columns;
    if (index >= count) return;
    const std::size_t row = index / columns;
    const std::size_t column = index % columns;
    const float value = static_cast<float>(accumulator[index]) *
                        activation_scales[row] * weight_scales[column];
    output[index] = __float2half_rn(value);
}

[[nodiscard]] std::shared_ptr<MatmulDescriptors> create_int8_descriptors(
    std::size_t m, std::size_t n, std::size_t k, std::size_t workspace_bytes,
    bool require_tensor_cores) {
    auto descriptors = std::make_shared<MatmulDescriptors>();
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &descriptors->operation, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t trans_a = CUBLAS_OP_N;
    const cublasOperation_t trans_b = CUBLAS_OP_T;
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        descriptors->operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        descriptors->operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->a, CUDA_R_8I, m, k, static_cast<std::int64_t>(k)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->b, CUDA_R_8I, n, k, static_cast<std::int64_t>(k)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->c, CUDA_R_32I, m, n, static_cast<std::int64_t>(n)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->d, CUDA_R_32I, m, n, static_cast<std::int64_t>(n)));
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&descriptors->preference));
    SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        descriptors->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    if (require_tensor_cores) {
        const std::uint64_t implementation_mask =
            static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA) |
            static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I) |
            static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I);
        SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            descriptors->preference, CUBLASLT_MATMUL_PREF_IMPL_MASK,
            &implementation_mask, sizeof(implementation_mask)));
    }
    return descriptors;
}

[[nodiscard]] bool is_verified_imma_algorithm(
    const cublasLtMatmulAlgo_t& algorithm) noexcept {
    std::uint64_t flags = 0;
    std::size_t written = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(
        &algorithm, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
        &flags, sizeof(flags), &written);
    if (status != CUBLAS_STATUS_SUCCESS || written != sizeof(flags)) return false;
    const std::uint64_t required =
        static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA) |
        static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I) |
        static_cast<std::uint64_t>(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I);
    return (flags & required) == required;
}

[[nodiscard]] Tensor upload_scale(const Runtime& runtime,
                                  const TensorView& scale_source,
                                  std::size_t rows) {
    TensorView scale = scale_source.squeeze_trailing_ones();
    if (scale.element_count() != rows) {
        throw CudaError("INT8 weight_scale must have one value per output row");
    }
    Tensor destination = Tensor::allocate_persistent(
        runtime, {rows}, ScalarType::Float32, TensorRole::QuantizationMetadata);
    const std::size_t span = source_span_bytes(scale);
    unsigned char* staging = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&staging), span, runtime.stream()));
    SDXL_CUDA_CHECK(cudaMemcpyAsync(staging, scale.data, span,
                                    cudaMemcpyHostToDevice, runtime.stream()));
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((rows + threads - 1) / threads);
    pack_f32_kernel<<<blocks, threads, 0, runtime.stream()>>>(
        staging, destination.float_data(), rows, pack_metadata(scale));
    SDXL_CUDA_CHECK(cudaGetLastError());
    SDXL_CUDA_CHECK(cudaFreeAsync(staging, runtime.stream()));
    return destination;
}

} // namespace

bool valid_convrot_group_size(std::size_t group_size) noexcept {
    if (group_size < 4 || group_size > 256) return false;
    while (group_size > 1) {
        if (group_size % 4 != 0) return false;
        group_size /= 4;
    }
    return true;
}

Tensor upload_or_quantize_int8_weight(
    const Runtime& runtime,
    const TensorView& source,
    const QuantizationMetadata* metadata,
    TensorRole role,
    const INT8WeightLoadOptions& options) {
    if (source.shape.size() != 2) {
        throw CudaError("native INT8 weight must be rank 2: " + source.source_key);
    }
    const std::size_t rows = static_cast<std::size_t>(source.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(source.shape[1]);
    if (rows == 0 || columns == 0) throw CudaError("native INT8 weight cannot be empty");

    const bool prequantized = source.dtype == DType::I8;
    if (options.require_prequantized && !prequantized) {
        throw CudaError("INT8 prequantized-only profile received a floating-point linear weight: " +
                        source.source_key);
    }
    if (!prequantized && !options.quantize_floating_weights) {
        throw CudaError("on-the-fly INT8 quantization is disabled for: " + source.source_key);
    }

    bool use_convrot = options.enable_convrot && columns % options.convrot_group_size == 0;
    std::size_t group_size = use_convrot ? options.convrot_group_size : 0;
    if (prequantized) {
        if (metadata == nullptr || !metadata->weight_scale.has_value()) {
            throw CudaError("prequantized INT8 weight is missing weight_scale: " + source.source_key);
        }
        use_convrot = metadata->convrot;
        group_size = use_convrot ? metadata->convrot_group_size : 0;
        if (use_convrot && (!valid_convrot_group_size(group_size) || columns % group_size != 0)) {
            throw CudaError("prequantized ConvRot metadata is incompatible with weight shape: " +
                            source.source_key);
        }
        if (metadata->weight_scale->element_count() != rows) {
            throw CudaError("prequantized INT8 weight_scale is not per-output-row: " +
                            source.source_key);
        }
    } else if (use_convrot && !valid_convrot_group_size(group_size)) {
        throw CudaError("INT8 ConvRot group size must be a power of four between 4 and 256");
    }

    Tensor weight = Tensor::allocate_persistent(
        runtime, {rows, columns}, ScalarType::Int8, role);
    Tensor scales;

    const std::size_t span = source_span_bytes(source);
    unsigned char* staging = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&staging), span, runtime.stream()));
    SDXL_CUDA_CHECK(cudaMemcpyAsync(staging, source.data, span,
                                    cudaMemcpyHostToDevice, runtime.stream()));
    constexpr unsigned threads = 256;

    if (prequantized) {
        const unsigned blocks = static_cast<unsigned>(
            (weight.elements() + threads - 1) / threads);
        pack_i8_kernel<<<blocks, threads, 0, runtime.stream()>>>(
            staging, weight.int8_data(), weight.elements(), pack_metadata(source));
        SDXL_CUDA_CHECK(cudaGetLastError());
        scales = upload_scale(runtime, *metadata->weight_scale, rows);
    } else {
        scales = Tensor::allocate_persistent(
            runtime, {rows}, ScalarType::Float32, TensorRole::QuantizationMetadata);
        const std::size_t shared_bytes = use_convrot ? 2 * group_size * sizeof(float) : 0;
        weight_amax_kernel<<<static_cast<unsigned>(rows), threads, shared_bytes,
                             runtime.stream()>>>(
            staging, scales.float_data(), rows, columns, group_size, pack_metadata(source));
        SDXL_CUDA_CHECK(cudaGetLastError());
        quantize_weight_kernel<<<static_cast<unsigned>(rows), threads, shared_bytes,
                                 runtime.stream()>>>(
            staging, weight.int8_data(), scales.float_data(), rows, columns,
            group_size, pack_metadata(source));
        SDXL_CUDA_CHECK(cudaGetLastError());
    }

    SDXL_CUDA_CHECK(cudaFreeAsync(staging, runtime.stream()));
    weight.attach_dequant_scale(scales, FP8ScaleMode::PerOutputChannel);
    if (use_convrot) weight.set_convrot_group_size(group_size);
    return weight;
}

Tensor int8_convrot_linear(const Runtime& runtime,
                           const Tensor& input,
                           const Tensor& weight) {
    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        input.rank() < 2 || weight.type() != ScalarType::Int8 || weight.rank() != 2 ||
        !weight.has_dequant_scale() ||
        weight.dequant_scale_mode() != FP8ScaleMode::PerOutputChannel) {
        throw CudaError("native INT8 linear received incompatible tensors");
    }
    const std::size_t k = input.shape().back();
    const std::size_t n = weight.size(0);
    const std::size_t m = input.elements() / k;
    if (weight.size(1) != k || weight.dequant_scale_count() != n) {
        throw CudaError("native INT8 linear feature or scale mismatch");
    }
    const std::size_t group_size = weight.convrot_group_size();
    if (group_size != 0 && (!valid_convrot_group_size(group_size) || k % group_size != 0)) {
        throw CudaError("native INT8 linear has invalid ConvRot group metadata");
    }
    if (k % 4 != 0) {
        throw CudaError("native INT8 Tensor Core linear requires K divisible by 4");
    }

    Tensor quantized_input = Tensor::allocate(
        runtime, {m, k}, ScalarType::Int8, TensorRole::Model);
    Tensor activation_scales = Tensor::allocate(
        runtime, {m}, ScalarType::Float32, TensorRole::QuantizationMetadata);
    const std::size_t shared_bytes = group_size == 0
        ? 0 : 2 * group_size * sizeof(float);
    constexpr unsigned threads = 256;
    if (input.type() == ScalarType::Float16) {
        activation_amax_kernel<<<static_cast<unsigned>(m), threads, shared_bytes,
                                 runtime.stream()>>>(
            input.half_data(), activation_scales.float_data(), m, k, group_size);
        SDXL_CUDA_CHECK(cudaGetLastError());
        quantize_activation_kernel<<<static_cast<unsigned>(m), threads, shared_bytes,
                                     runtime.stream()>>>(
            input.half_data(), quantized_input.int8_data(), activation_scales.float_data(),
            m, k, group_size);
    } else {
        activation_amax_kernel<<<static_cast<unsigned>(m), threads, shared_bytes,
                                 runtime.stream()>>>(
            input.float_data(), activation_scales.float_data(), m, k, group_size);
        SDXL_CUDA_CHECK(cudaGetLastError());
        quantize_activation_kernel<<<static_cast<unsigned>(m), threads, shared_bytes,
                                     runtime.stream()>>>(
            input.float_data(), quantized_input.int8_data(), activation_scales.float_data(),
            m, k, group_size);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());

    Tensor accumulator = Tensor::allocate(
        runtime, {m, n}, ScalarType::Int32, TensorRole::Model);
    runtime.state()->int8_linear_calls.fetch_add(1, std::memory_order_relaxed);

    MatmulPlan plan{};
    const MatmulKey key{m, n, k, 801};
    bool used_cublas_lt_imma = false;
    bool execution_failed = false;
    {
        std::lock_guard lock(runtime.state()->cublas_mutex);
        const auto cached = runtime.state()->matmul_plans.find(key);
        if (cached != runtime.state()->matmul_plans.end()) {
            plan = cached->second;
        } else {
            try {
                plan.descriptors = create_int8_descriptors(
                    m, n, k, runtime.options().cublas_workspace_bytes,
                    runtime.options().int8_require_tensor_cores);

                constexpr int maximum_heuristics = 32;
                std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
                    maximum_heuristics);
                int returned = 0;
                const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
                    runtime.cublas_lt(), plan.descriptors->operation,
                    plan.descriptors->a, plan.descriptors->b,
                    plan.descriptors->c, plan.descriptors->d,
                    plan.descriptors->preference, maximum_heuristics,
                    heuristics.data(), &returned);

                bool found = false;
                if (heuristic_status == CUBLAS_STATUS_SUCCESS) {
                    for (int index = 0; index < returned; ++index) {
                        const auto& heuristic = heuristics[static_cast<std::size_t>(index)];
                        if (heuristic.state != CUBLAS_STATUS_SUCCESS ||
                            !is_verified_imma_algorithm(heuristic.algo)) {
                            continue;
                        }
                        plan.algorithm = heuristic.algo;
                        plan.workspace_bytes = heuristic.workspaceSize;
                        found = true;
                        break;
                    }
                }
                if (!found) plan = {};
            } catch (const CudaError&) {
                plan = {};
            }
            runtime.state()->matmul_plans.emplace(key, plan);
        }

        if (plan.descriptors != nullptr) {
            const std::int32_t alpha = 1;
            const std::int32_t beta = 0;
            const cublasStatus_t status = cublasLtMatmul(
                runtime.cublas_lt(), plan.descriptors->operation, &alpha,
                quantized_input.data(), plan.descriptors->a,
                weight.data(), plan.descriptors->b, &beta,
                accumulator.data(), plan.descriptors->c,
                accumulator.data(), plan.descriptors->d,
                &plan.algorithm, runtime.state()->cublas_workspace,
                plan.workspace_bytes, runtime.stream());
            used_cublas_lt_imma = status == CUBLAS_STATUS_SUCCESS;
            execution_failed = !used_cublas_lt_imma;
            if (execution_failed) {
                runtime.state()->matmul_plans[key] = MatmulPlan{};
            }
        }
    }

    if (used_cublas_lt_imma) {
        runtime.state()->int8_cublaslt_imma_calls.fetch_add(
            1, std::memory_order_relaxed);
    } else {
        if (execution_failed) {
            runtime.state()->int8_tensor_core_execution_failures.fetch_add(
                1, std::memory_order_relaxed);
        } else {
            runtime.state()->int8_tensor_core_plan_misses.fetch_add(
                1, std::memory_order_relaxed);
        }

        if (runtime.options().int8_require_tensor_cores) {
            std::ostringstream message;
            message << "INT8 Tensor Core-only execution failed for Linear shape M="
                    << m << ", N=" << n << ", K=" << k << ": "
                    << (execution_failed
                        ? "verified cuBLASLt IMMA plan failed at execution"
                        : "no verified cuBLASLt IMMA heuristic plan was available")
                    << ". Strict/tensorcore profiles never fall back to DP4A.";
            throw CudaError(message.str());
        }

        runtime.state()->int8_dp4a_fallback_calls.fetch_add(
            1, std::memory_order_relaxed);
        const dim3 threads2d(16, 16);
        const dim3 blocks2d(
            static_cast<unsigned>((n + threads2d.x - 1) / threads2d.x),
            static_cast<unsigned>((m + threads2d.y - 1) / threads2d.y));
        int8_gemm_dp4a_kernel<<<blocks2d, threads2d, 0, runtime.stream()>>>(
            quantized_input.int8_data(), weight.int8_data(), accumulator.int32_data(),
            m, n, k);
        SDXL_CUDA_CHECK(cudaGetLastError());
    }

    std::vector<std::size_t> output_shape = input.shape();
    output_shape.back() = n;
    Tensor output = Tensor::allocate(runtime, output_shape, input.type(), input.role());
    const std::size_t count = m * n;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    if (output.type() == ScalarType::Float16) {
        dequantize_gemm_f16_kernel<<<blocks, threads, 0, runtime.stream()>>>(
            accumulator.int32_data(), output.half_data(), activation_scales.float_data(),
            weight.dequant_scale_data(), m, n);
    } else {
        dequantize_gemm_f32_kernel<<<blocks, threads, 0, runtime.stream()>>>(
            accumulator.int32_data(), output.float_data(), activation_scales.float_data(),
            weight.dequant_scale_data(), m, n);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace sdxl::cuda
