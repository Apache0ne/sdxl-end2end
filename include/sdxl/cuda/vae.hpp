#pragma once

#include "sdxl/cuda/ops.hpp"
#include "sdxl/cuda/weights.hpp"
#include "sdxl/sdxl.hpp"

#include <cstddef>
#include <string_view>

namespace sdxl::cuda {

struct VAEOptions {
    float scaling_factor = 0.13025F;
    std::size_t norm_groups = 32;
    bool check_finite_output = false;
};

class VAE final {
public:
    VAE(const Runtime& runtime,
        const SDXLModel& model,
        const WeightStore& weights,
        VAEOptions options = {});

    // Decodes SDXL latent tensors [B,4,H,W] into normalized RGB tensors
    // [B,3,H*8,W*8] in FP32 (SDXL force_upcast behavior). The decoder output remains in the model's native
    // approximately [-1,1] range; ImageConverter performs clamping and u8 conversion.
    [[nodiscard]] Tensor decode(const Tensor& latents) const;

private:
    [[nodiscard]] Tensor resnet_block(std::string_view prefix,
                                      const Tensor& input) const;
    [[nodiscard]] Tensor attention_block(std::string_view prefix,
                                         const Tensor& input) const;

    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    const WeightStore* weights_ = nullptr;
    Ops ops_;
    VAEOptions options_;
};

} // namespace sdxl::cuda
