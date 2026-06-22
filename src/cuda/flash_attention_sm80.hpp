#pragma once

#include "sdxl/cuda/tensor.hpp"

#include <cstddef>

namespace sdxl::cuda {

// Launches the in-tree forward-only tiled attention kernel for contiguous
// [batch, sequence, heads * 64] FP16 tensors. The kernel uses Tensor Core WMMA
// for QK^T, online softmax, and tiled V accumulation without materializing the
// complete QxK score matrix.
void launch_flash_attention_sm80_hdim64(const Runtime& runtime,
                                        const Tensor& query,
                                        const Tensor& key,
                                        const Tensor& value,
                                        const Tensor* key_mask,
                                        Tensor& output,
                                        std::size_t heads,
                                        bool causal);

} // namespace sdxl::cuda
