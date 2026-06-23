#pragma once

#include "sdxl/cuda/tensor.hpp"
#include "sdxl/cuda/weights.hpp"
#include "sdxl/model.hpp"

namespace sdxl::cuda {

[[nodiscard]] bool valid_convrot_group_size(std::size_t group_size) noexcept;

[[nodiscard]] Tensor upload_or_quantize_int8_weight(
    const Runtime& runtime,
    const TensorView& source,
    const QuantizationMetadata* metadata,
    TensorRole role,
    const INT8WeightLoadOptions& options);

[[nodiscard]] Tensor int8_convrot_linear(
    const Runtime& runtime,
    const Tensor& input,
    const Tensor& weight);

} // namespace sdxl::cuda
