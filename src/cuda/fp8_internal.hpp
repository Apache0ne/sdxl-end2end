#pragma once

#include "sdxl/cuda/tensor.hpp"

namespace sdxl::cuda {

[[nodiscard]] Tensor quantize_tensor_fp8(const Runtime& runtime,
                                         const Tensor& source_f16,
                                         ScalarType destination_type);

[[nodiscard]] Tensor fp8_weight_only_linear(const Runtime& runtime,
                                                    const Tensor& input_f16,
                                                    const Tensor& weight_fp8);

} // namespace sdxl::cuda
