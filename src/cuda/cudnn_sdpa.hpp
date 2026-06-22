#pragma once

#include "sdxl/cuda/tensor.hpp"

#include <cstddef>

namespace sdxl::cuda {

[[nodiscard]] bool cudnn_frontend_sdpa_compiled() noexcept;

// Executes cuDNN Frontend's FP16 SDPA graph directly on the runtime stream.
// Q/K/V/O use the engine's contiguous BSHD physical layout represented to
// cuDNN as logical [B,H,S,D] tensors with explicit strides.
void launch_cudnn_frontend_sdpa(const Runtime& runtime,
                                const Tensor& query,
                                const Tensor& key,
                                const Tensor& value,
                                Tensor& output,
                                std::size_t heads,
                                bool causal);

} // namespace sdxl::cuda
