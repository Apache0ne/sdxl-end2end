#pragma once

#include "sdxl/cuda/ops.hpp"
#include "sdxl/unet.hpp"

namespace sdxl::cuda {

struct UNetOptions {
    bool check_finite_output = false;
};

class UNet final {
public:
    UNet(const Runtime& runtime,
         const SDXLModel& model,
         const WeightStore& weights,
         UNetOptions options = {});

    [[nodiscard]] Tensor forward(const Tensor& latent_sample,
                                 float timestep,
                                 const Tensor& encoder_hidden_states,
                                 const Tensor& pooled_text_embeds,
                                 const Tensor& time_ids) const;

private:
    [[nodiscard]] Tensor resnet_block(std::string_view prefix,
                                      const Tensor& input,
                                      const Tensor& temb) const;
    [[nodiscard]] Tensor transformer_block(std::string_view prefix,
                                           const Tensor& input,
                                           const Tensor& encoder_hidden_states,
                                           std::size_t heads) const;
    [[nodiscard]] Tensor transformer2d(std::string_view prefix,
                                      const Tensor& input,
                                      const Tensor& encoder_hidden_states,
                                      std::size_t depth,
                                      std::size_t heads) const;
    [[nodiscard]] Tensor build_time_embedding(const Tensor& latents,
                                              float timestep,
                                              const Tensor& pooled_text_embeds,
                                              const Tensor& time_ids) const;

    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    const WeightStore* weights_ = nullptr;
    Ops ops_;
    UNetOptions options_;
};

} // namespace sdxl::cuda
