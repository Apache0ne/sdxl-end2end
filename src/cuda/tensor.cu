#include "sdxl/cuda/tensor.hpp"
#include "runtime_internal.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace sdxl::cuda {
namespace {

[[nodiscard]] std::uint16_t float_to_half_bits(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    std::uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0xFFU) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7C00U : 0x7E00U));
    }
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000U;
        const unsigned shift = static_cast<unsigned>(14 - adjusted);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U) != 0U)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400U) {
            rounded_mantissa = 0;
            if (adjusted + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
        }
    }
    return static_cast<std::uint16_t>(sign |
        (static_cast<std::uint32_t>(adjusted) << 10U) | rounded_mantissa);
}

[[nodiscard]] float half_bits_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa = bits & 0x03FFU;
    std::uint32_t output = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            output = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --unbiased;
            }
            mantissa &= 0x03FFU;
            output = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23U) |
                     (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        output = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        output = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(output);
}

[[nodiscard]] float decode_fp8_byte(std::uint8_t bits, bool e5m2) noexcept {
    const bool negative = (bits & 0x80U) != 0;
    const unsigned mantissa_bits = e5m2 ? 2U : 3U;
    const unsigned exponent_bits = e5m2 ? 5U : 4U;
    const int bias = e5m2 ? 15 : 7;
    const unsigned mantissa_mask = (1U << mantissa_bits) - 1U;
    const unsigned exponent_mask = (1U << exponent_bits) - 1U;
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    const unsigned mantissa = bits & mantissa_mask;
    float value = 0.0F;
    if (exponent == 0U) {
        if (mantissa != 0U) {
            value = std::ldexp(static_cast<float>(mantissa) /
                                   static_cast<float>(1U << mantissa_bits),
                               1 - bias);
        }
    } else if (exponent == exponent_mask) {
        if (!e5m2 && mantissa != mantissa_mask) {
            value = std::ldexp(1.0F + static_cast<float>(mantissa) /
                                           static_cast<float>(1U << mantissa_bits),
                               static_cast<int>(exponent) - bias);
        } else if (mantissa == 0U && e5m2) {
            value = std::numeric_limits<float>::infinity();
        } else {
            value = std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) /
                                   static_cast<float>(1U << mantissa_bits),
                           static_cast<int>(exponent) - bias);
    }
    return negative ? -value : value;
}

void validate_allocation_policy(const Runtime& runtime, ScalarType type, TensorRole role) {
    if (type != ScalarType::Float32 || !runtime.options().strict_non_vae_fp32) return;
    if (role == TensorRole::SamplerState || role == TensorRole::VAE ||
        role == TensorRole::FP8ScaleMetadata || role == TensorRole::QuantizationMetadata ||
        role == TensorRole::HostInterop) return;
    throw CudaError("strict precision policy rejected a non-VAE float32 tensor allocation");
}

} // namespace

bool is_fp8(ScalarType type) noexcept {
    return type == ScalarType::Float8E4M3 || type == ScalarType::Float8E5M2;
}

std::size_t scalar_size(ScalarType type) noexcept {
    switch (type) {
    case ScalarType::Float8E4M3:
    case ScalarType::Float8E5M2: return sizeof(std::uint8_t);
    case ScalarType::Int8: return sizeof(std::int8_t);
    case ScalarType::Float16: return sizeof(__half);
    case ScalarType::Float32: return sizeof(float);
    case ScalarType::Int32: return sizeof(std::int32_t);
    }
    return 0;
}

const char* scalar_name(ScalarType type) noexcept {
    switch (type) {
    case ScalarType::Float8E4M3: return "float8_e4m3";
    case ScalarType::Float8E5M2: return "float8_e5m2";
    case ScalarType::Int8: return "int8";
    case ScalarType::Float16: return "float16";
    case ScalarType::Float32: return "float32";
    case ScalarType::Int32: return "int32";
    }
    return "unknown";
}

const char* tensor_role_name(TensorRole role) noexcept {
    switch (role) {
    case TensorRole::Model: return "model";
    case TensorRole::SamplerState: return "sampler_state";
    case TensorRole::VAE: return "vae";
    case TensorRole::FP8ScaleMetadata: return "fp8_scale_metadata";
    case TensorRole::QuantizationMetadata: return "quantization_metadata";
    case TensorRole::HostInterop: return "host_interop";
    }
    return "unknown";
}

DeviceAllocation::~DeviceAllocation() {
    if (pointer == nullptr || runtime == nullptr) return;
    cudaSetDevice(runtime->device);
    if (persistent) runtime->release_persistent_block(pointer, bytes);
    else runtime->release_block(pointer, bytes);
}

std::size_t element_count(std::span<const std::size_t> shape) {
    if (shape.empty()) return 0;
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) return 0;
        if (count > static_cast<std::size_t>(-1) / dimension) {
            throw CudaError("tensor element count overflow");
        }
        count *= dimension;
    }
    return count;
}

Tensor::Tensor(std::shared_ptr<DeviceAllocation> allocation,
               std::shared_ptr<RuntimeState> runtime,
               void* data,
               std::vector<std::size_t> shape,
               ScalarType type,
               TensorRole role)
    : allocation_(std::move(allocation)),
      runtime_(std::move(runtime)),
      data_(data),
      shape_(std::move(shape)),
      type_(type),
      role_(role) {}

Tensor Tensor::allocate(const Runtime& runtime, std::vector<std::size_t> shape,
                        ScalarType type, TensorRole role) {
    validate_allocation_policy(runtime, type, role);
    const std::size_t bytes = element_count(shape) * scalar_size(type);
    if (bytes == 0) throw CudaError("cannot allocate an empty CUDA tensor");
    auto allocation = std::make_shared<DeviceAllocation>();
    allocation->runtime = runtime.state();
    allocation->bytes = bytes;
    allocation->pointer = runtime.state()->acquire_block(bytes);
    return Tensor(allocation, runtime.state(), allocation->pointer, std::move(shape), type, role);
}

Tensor Tensor::allocate_persistent(const Runtime& runtime,
                                   std::vector<std::size_t> shape,
                                   ScalarType type, TensorRole role) {
    validate_allocation_policy(runtime, type, role);
    const std::size_t bytes = element_count(shape) * scalar_size(type);
    if (bytes == 0) throw CudaError("cannot allocate an empty persistent CUDA tensor");
    auto allocation = std::make_shared<DeviceAllocation>();
    allocation->runtime = runtime.state();
    allocation->bytes = bytes;
    allocation->persistent = true;
    allocation->pointer = runtime.state()->acquire_persistent_block(bytes);
    return Tensor(allocation, runtime.state(), allocation->pointer,
                  std::move(shape), type, role);
}

Tensor Tensor::zeros_persistent(const Runtime& runtime,
                                std::vector<std::size_t> shape,
                                ScalarType type, TensorRole role) {
    Tensor result = allocate_persistent(runtime, std::move(shape), type, role);
    SDXL_CUDA_CHECK(cudaMemsetAsync(result.data(), 0, result.bytes(), runtime.stream()));
    return result;
}

Tensor Tensor::zeros(const Runtime& runtime, std::vector<std::size_t> shape,
                     ScalarType type, TensorRole role) {
    Tensor result = allocate(runtime, std::move(shape), type, role);
    SDXL_CUDA_CHECK(cudaMemsetAsync(result.data(), 0, result.bytes(), runtime.stream()));
    return result;
}

Tensor Tensor::from_host_f32(const Runtime& runtime, const FloatTensor& host,
                            ScalarType type, TensorRole role) {
    if (host.values.size() != element_count(host.shape)) {
        throw CudaError("host FloatTensor shape does not match its storage");
    }
    Tensor result = allocate(runtime, host.shape, type, role);
    if (type == ScalarType::Float32) {
        SDXL_CUDA_CHECK(cudaMemcpyAsync(result.data(), host.values.data(), result.bytes(),
                                        cudaMemcpyHostToDevice, runtime.stream()));
    } else if (type == ScalarType::Float16) {
        // Convert on the host so the CUDA model path never allocates a float32 staging tensor.
        std::vector<std::uint16_t> packed(host.values.size());
        std::transform(host.values.begin(), host.values.end(), packed.begin(), float_to_half_bits);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(result.data(), packed.data(), result.bytes(),
                                        cudaMemcpyHostToDevice, runtime.stream()));
    } else {
        throw CudaError("from_host_f32 only supports float16 and float32 targets");
    }
    return result;
}

Tensor Tensor::from_host_i32(const Runtime& runtime,
                             std::vector<std::size_t> shape,
                             std::span<const std::int32_t> values) {
    if (element_count(shape) != values.size()) throw CudaError("int32 host tensor shape mismatch");
    Tensor result = allocate(runtime, std::move(shape), ScalarType::Int32, TensorRole::Model);
    SDXL_CUDA_CHECK(cudaMemcpyAsync(result.data(), values.data(), result.bytes(),
                                    cudaMemcpyHostToDevice, runtime.stream()));
    return result;
}

Tensor Tensor::view(std::vector<std::size_t> shape, std::size_t element_offset) const {
    if (!defined()) throw CudaError("cannot view an undefined CUDA tensor");
    const std::size_t requested = element_count(shape);
    if (element_offset + requested > elements()) throw CudaError("CUDA tensor view exceeds storage");
    auto* bytes = static_cast<std::byte*>(data_);
    Tensor result(allocation_, runtime_, bytes + element_offset * scalar_size(type_),
                  std::move(shape), type_, role_);
    result.dequant_scale_allocation_ = dequant_scale_allocation_;
    result.dequant_scale_data_ = dequant_scale_data_;
    result.dequant_scale_count_ = dequant_scale_count_;
    result.dequant_scale_mode_ = dequant_scale_mode_;
    result.fp8_storage_layout_ = fp8_storage_layout_;
    result.convrot_group_size_ = convrot_group_size_;
    return result;
}

Tensor Tensor::reshape(std::vector<std::size_t> shape) const {
    if (element_count(shape) != elements()) throw CudaError("CUDA tensor reshape changes element count");
    return view(std::move(shape), 0);
}

std::size_t Tensor::elements() const noexcept {
    if (shape_.empty()) return 0;
    std::size_t count = 1;
    for (const std::size_t dimension : shape_) count *= dimension;
    return count;
}

FloatTensor Tensor::to_host_f32(const Runtime& runtime) const {
    if (!defined()) throw CudaError("cannot download an undefined CUDA tensor");
    FloatTensor result;
    result.shape = shape_;
    result.values.resize(elements());
    if (type_ == ScalarType::Float32) {
        SDXL_CUDA_CHECK(cudaMemcpyAsync(result.values.data(), data(), bytes(),
                                        cudaMemcpyDeviceToHost, runtime.stream()));
        runtime.synchronize();
        return result;
    }
    if (type_ == ScalarType::Float16) {
        std::vector<std::uint16_t> packed(elements());
        SDXL_CUDA_CHECK(cudaMemcpyAsync(packed.data(), data(), bytes(),
                                        cudaMemcpyDeviceToHost, runtime.stream()));
        runtime.synchronize();
        std::transform(packed.begin(), packed.end(), result.values.begin(), half_bits_to_float);
        return result;
    }
    if (is_fp8(type_)) {
        std::vector<std::uint8_t> packed(elements());
        std::vector<float> scales(dequant_scale_count_ == 0 ? 1 : dequant_scale_count_, 1.0F);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(packed.data(), data(), bytes(),
                                        cudaMemcpyDeviceToHost, runtime.stream()));
        if (dequant_scale_data_ != nullptr) {
            SDXL_CUDA_CHECK(cudaMemcpyAsync(scales.data(), dequant_scale_data_,
                                            scales.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost, runtime.stream()));
        }
        runtime.synchronize();
        const bool e5m2 = type_ == ScalarType::Float8E5M2;
        const std::size_t row_width = rank() == 2 ? size(1) : elements();
        for (std::size_t logical = 0; logical < elements(); ++logical) {
            std::size_t physical = logical;
            std::size_t output_channel = row_width == 0 ? 0 : logical / row_width;
            if (rank() == 2 && fp8_storage_layout_ == FP8StorageLayout::KMajorKN) {
                const std::size_t k = logical % size(1);
                physical = k * size(0) + output_channel;
            }
            std::size_t scale_index = 0;
            if (dequant_scale_mode_ == FP8ScaleMode::PerOutputChannel && !scales.empty()) {
                scale_index = std::min(output_channel, scales.size() - 1);
            }
            result.values[logical] = decode_fp8_byte(packed[physical], e5m2) * scales[scale_index];
        }
        return result;
    }
    if (type_ == ScalarType::Int8) {
        std::vector<std::int8_t> packed(elements());
        std::vector<float> scales(dequant_scale_count_ == 0 ? 1 : dequant_scale_count_, 1.0F);
        SDXL_CUDA_CHECK(cudaMemcpyAsync(packed.data(), data(), bytes(),
                                        cudaMemcpyDeviceToHost, runtime.stream()));
        if (dequant_scale_data_ != nullptr) {
            SDXL_CUDA_CHECK(cudaMemcpyAsync(scales.data(), dequant_scale_data_,
                                            scales.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost, runtime.stream()));
        }
        runtime.synchronize();
        const std::size_t row_width = rank() == 2 ? size(1) : elements();
        for (std::size_t logical = 0; logical < elements(); ++logical) {
            const std::size_t output_channel = row_width == 0 ? 0 : logical / row_width;
            std::size_t scale_index = 0;
            if (dequant_scale_mode_ == FP8ScaleMode::PerOutputChannel && !scales.empty()) {
                scale_index = std::min(output_channel, scales.size() - 1);
            }
            result.values[logical] = static_cast<float>(packed[logical]) * scales[scale_index];
        }
        return result;
    }
    throw CudaError("cannot convert this integer CUDA tensor to FloatTensor");
}

void Tensor::set_fp8_storage_layout(FP8StorageLayout layout) {
    if (!is_fp8(type_) || rank() != 2) {
        throw CudaError("FP8 storage layout can only be set on rank-2 FP8 tensors");
    }
    fp8_storage_layout_ = layout;
}

void Tensor::set_convrot_group_size(std::size_t group_size) {
    if (type_ != ScalarType::Int8 || rank() != 2) {
        throw CudaError("ConvRot metadata can only be attached to rank-2 INT8 tensors");
    }
    if (group_size != 0) {
        std::size_t value = group_size;
        if (value < 4 || value > 256) {
            throw CudaError("ConvRot group size must be zero or a power of four from 4 to 256");
        }
        while (value > 1) {
            if (value % 4 != 0) {
                throw CudaError("ConvRot group size must be zero or a power of four from 4 to 256");
            }
            value /= 4;
        }
    }
    convrot_group_size_ = group_size;
}

void Tensor::attach_dequant_scale(const Tensor& scale, FP8ScaleMode mode) {
    if (!is_fp8(type_) && type_ != ScalarType::Int8) {
        throw CudaError("dequantization scales can only be attached to FP8 or INT8 tensors");
    }
    if (scale.type_ != ScalarType::Float32 ||
        (scale.role_ != TensorRole::FP8ScaleMetadata &&
         scale.role_ != TensorRole::QuantizationMetadata)) {
        throw CudaError("dequantization scale must be float32 quantization metadata");
    }
    const std::size_t expected = mode == FP8ScaleMode::PerOutputChannel
        ? (rank() == 2 ? size(0) : 0) : 1;
    if (mode == FP8ScaleMode::None || expected == 0 || scale.elements() != expected) {
        throw CudaError("FP8 dequantization scale shape does not match its scale mode");
    }
    validate_same_runtime(*this, scale);
    dequant_scale_allocation_ = scale.allocation_;
    dequant_scale_data_ = scale.float_data();
    dequant_scale_count_ = scale.elements();
    dequant_scale_mode_ = mode;
}

void Tensor::copy_from(const Runtime& runtime, const Tensor& source) {
    validate_same_runtime(*this, source);
    if (shape_ != source.shape_ || type_ != source.type_ || role_ != source.role_) {
        throw CudaError("CUDA tensor copy shape/type/role mismatch");
    }
    SDXL_CUDA_CHECK(cudaMemcpyAsync(data(), source.data(), bytes(),
                                    cudaMemcpyDeviceToDevice, runtime.stream()));
    if (is_fp8(type_) || type_ == ScalarType::Int8) {
        dequant_scale_allocation_ = source.dequant_scale_allocation_;
        dequant_scale_data_ = source.dequant_scale_data_;
        dequant_scale_count_ = source.dequant_scale_count_;
        dequant_scale_mode_ = source.dequant_scale_mode_;
        fp8_storage_layout_ = source.fp8_storage_layout_;
        convrot_group_size_ = source.convrot_group_size_;
    }
}

void validate_same_runtime(const Tensor& first, const Tensor& second) {
    if (!first.defined() || !second.defined()) throw CudaError("operation received an undefined CUDA tensor");
    if (first.runtime_state() != second.runtime_state()) throw CudaError("CUDA tensors belong to different runtimes");
}

} // namespace sdxl::cuda
