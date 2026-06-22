#pragma once

#include "sdxl/cuda/text_encoder.hpp"
#include "sdxl/cuda/unet.hpp"
#include "sdxl/denoiser.hpp"

#include <optional>
#include <vector>

namespace sdxl::cuda {

struct DenoiseOptions {
    std::size_t width = 1024;
    std::size_t height = 1024;
    std::size_t inference_steps = 30;
    float guidance_scale = 5.0F;
    float guidance_rescale = 0.0F;
    float ddim_eta = 0.0F;
    std::uint64_t seed = 0;
    std::vector<std::uint64_t> batch_seeds;
    SchedulerKind scheduler = SchedulerKind::EulerDiscrete;
    SchedulerConfig scheduler_config{};
    SDXLMicroConditioning positive_micro{};
    std::optional<SDXLMicroConditioning> negative_micro;
    bool profile_steps = false;
};

struct DenoiseResult {
    Tensor latents;
    std::vector<float> timesteps;
    std::vector<float> step_milliseconds;
};

class Denoiser final {
public:
    Denoiser(const Runtime& runtime,
             const SDXLModel& model,
             const WeightStore& weights,
             UNetOptions options = {});

    [[nodiscard]] DenoiseResult denoise(const ClassifierFreeConditioning& conditioning,
                                        DenoiseOptions options = {}) const;

    // Graph/persistent-session helpers. The supplied latent and time-ID buffers
    // are fixed-address external inputs; all loop intermediates remain internal.
    [[nodiscard]] Tensor create_initial_latents(const DenoiseOptions& options,
                                                std::size_t batch) const;
    [[nodiscard]] Tensor create_time_ids(std::size_t batch,
                                         const DenoiseOptions& options,
                                         bool classifier_free = true) const;
    void denoise_into(const ClassifierFreeConditioning& conditioning,
                      const Tensor& initial_latents,
                      const Tensor& time_ids,
                      Tensor& output_latents,
                      DenoiseOptions options = {}) const;

private:
    void validate_inputs(const ClassifierFreeConditioning& conditioning,
                         const DenoiseOptions& options) const;
    [[nodiscard]] DenoiseResult run_loop(const ClassifierFreeConditioning& conditioning,
                                         Tensor latents,
                                         const Tensor& time_ids,
                                         DenoiseOptions options) const;

    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    const WeightStore* weights_ = nullptr;
    Ops ops_;
    UNet unet_;
};

} // namespace sdxl::cuda
