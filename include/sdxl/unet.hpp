#pragma once

#include "sdxl/text_encoder.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace sdxl {

struct SDXLMicroConditioning {
    std::uint32_t original_height = 0;
    std::uint32_t original_width = 0;
    std::uint32_t crop_top = 0;
    std::uint32_t crop_left = 0;
    std::uint32_t target_height = 0;
    std::uint32_t target_width = 0;

    [[nodiscard]] FloatTensor time_ids(std::size_t batch_size) const;
};

struct UNetExecutionOptions {
    std::size_t thread_count = 0;
    std::size_t linear_output_block = 16;
    std::size_t convolution_output_block = 1;
    bool check_finite_outputs = true;
    std::function<void(std::string_view stage,
                       std::size_t block,
                       std::size_t layer,
                       std::size_t total)> progress;
};

class SDXLUNet final {
public:
    explicit SDXLUNet(const SDXLModel& model, UNetExecutionOptions options = {});

    [[nodiscard]] FloatTensor forward(
        const FloatTensor& latent_sample,
        float timestep,
        const FloatTensor& encoder_hidden_states,
        const FloatTensor& pooled_text_embeds,
        const FloatTensor& time_ids) const;

private:
    const SDXLModel* model_ = nullptr;
    UNetExecutionOptions options_;
};

} // namespace sdxl
