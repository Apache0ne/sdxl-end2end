#pragma once

#include "sdxl/scheduler.hpp"
#include "sdxl/unet.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace sdxl {

struct SDXLDenoiseOptions {
    std::size_t width = 1024;
    std::size_t height = 1024;
    std::size_t inference_steps = 30;
    float guidance_scale = 5.0F;
    float guidance_rescale = 0.0F;
    float ddim_eta = 0.0F;
    std::uint64_t seed = 0;
    SamplerKind sampler = SamplerKind::DPMpp2M;
    SchedulerKind scheduler = SchedulerKind::Normal;
    SamplerConfig sampler_config{};
    SchedulerConfig scheduler_config{};
    SDXLMicroConditioning positive_micro{};
    std::optional<SDXLMicroConditioning> negative_micro;
    std::function<void(std::size_t step,
                       std::size_t total_steps,
                       float timestep,
                       const FloatTensor& latents)> progress;
};

struct SDXLDenoiseResult {
    FloatTensor latents;
    std::vector<float> timesteps;
};

class SDXLDenoiser final {
public:
    explicit SDXLDenoiser(const SDXLModel& model,
                          UNetExecutionOptions unet_options = {});

    [[nodiscard]] SDXLDenoiseResult denoise(
        const SDXLClassifierFreeConditioning& conditioning,
        SDXLDenoiseOptions options = {}) const;

    [[nodiscard]] SDXLDenoiseResult denoise_from_latents(
        const SDXLClassifierFreeConditioning& conditioning,
        FloatTensor initial_latents,
        SDXLDenoiseOptions options = {}) const;

private:
    const SDXLModel* model_ = nullptr;
    SDXLUNet unet_;
};

} // namespace sdxl
