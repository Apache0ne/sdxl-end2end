#pragma once

#include "sdxl/text_encoder.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace sdxl {

enum class PredictionType {
    Epsilon,
    Sample,
    VPrediction
};

struct SchedulerConfig {
    std::size_t training_timesteps = 1000;
    float beta_start = 0.00085F;
    float beta_end = 0.012F;
    PredictionType prediction_type = PredictionType::Epsilon;
};

class EulerDiscreteScheduler final {
public:
    explicit EulerDiscreteScheduler(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    [[nodiscard]] const std::vector<float>& sigmas() const noexcept { return sigmas_; }
    [[nodiscard]] float initial_noise_sigma() const;

    [[nodiscard]] FloatTensor scale_model_input(const FloatTensor& sample,
                                                std::size_t step_index) const;
    [[nodiscard]] FloatTensor step(const FloatTensor& model_output,
                                   std::size_t step_index,
                                   const FloatTensor& sample) const;

private:
    SchedulerConfig config_;
    std::vector<float> training_sigmas_;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

class DDIMScheduler final {
public:
    explicit DDIMScheduler(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    [[nodiscard]] float initial_noise_sigma() const noexcept { return 1.0F; }

    [[nodiscard]] FloatTensor scale_model_input(const FloatTensor& sample,
                                                std::size_t step_index) const;
    [[nodiscard]] FloatTensor step(const FloatTensor& model_output,
                                   std::size_t step_index,
                                   const FloatTensor& sample,
                                   float eta,
                                   std::mt19937_64* generator = nullptr) const;

private:
    SchedulerConfig config_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> timesteps_;
};

[[nodiscard]] FloatTensor random_normal_tensor(const std::vector<std::size_t>& shape,
                                               std::uint64_t seed);

} // namespace sdxl
