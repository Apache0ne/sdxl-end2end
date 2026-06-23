#pragma once

#include "sdxl/text_encoder.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace sdxl {

enum class PredictionType {
    Epsilon,
    Sample,
    VPrediction
};

enum class SamplerKind : std::uint8_t {
    DPMpp2M,
    DPMppSDE,
    Euler,
    EulerAncestral,
    DPMpp2SAncestralCFGpp,
    DDIM
};

enum class NoiseDevice : std::uint8_t { CPU, GPU };

enum class SamplerStatePrecision : std::uint8_t { Float16, Float32 };
enum class InitialNoiseScaling : std::uint8_t { Sigma, ComfyUIMaxDenoise };

enum class SchedulerKind : std::uint8_t {
    Normal,
    Karras,
    Exponential,
    SGMUniform,
    Simple,
    DDIMUniform,
    DDIMTrailing,
    Beta,
    LinearQuadratic,
    KLOptimal,
    GITS
};

[[nodiscard]] SamplerKind parse_sampler_kind(std::string value);
[[nodiscard]] SchedulerKind parse_scheduler_kind(std::string value);
[[nodiscard]] const char* sampler_kind_name(SamplerKind kind) noexcept;
[[nodiscard]] const char* scheduler_kind_name(SchedulerKind kind) noexcept;
[[nodiscard]] bool is_sampler_kind_name(std::string_view value) noexcept;
[[nodiscard]] bool is_scheduler_kind_name(std::string_view value) noexcept;
[[nodiscard]] NoiseDevice parse_noise_device(std::string value);
[[nodiscard]] const char* noise_device_name(NoiseDevice device) noexcept;
[[nodiscard]] SamplerStatePrecision parse_sampler_state_precision(std::string value);
[[nodiscard]] const char* sampler_state_precision_name(SamplerStatePrecision value) noexcept;
[[nodiscard]] InitialNoiseScaling parse_initial_noise_scaling(std::string value);
[[nodiscard]] const char* initial_noise_scaling_name(InitialNoiseScaling value) noexcept;

struct AncestralStep {
    float down = 0.0F;
    float up = 0.0F;
};

[[nodiscard]] AncestralStep make_ancestral_step(float sigma_from,
                                                 float sigma_to,
                                                 float eta);

struct DPMppSDEStage {
    float sigma_mid = 0.0F;
    AncestralStep midpoint{};
    AncestralStep final{};
    float midpoint_sample_coefficient = 0.0F;
    float midpoint_denoised_coefficient = 0.0F;
    float final_sample_coefficient = 0.0F;
    float final_denoised_coefficient = 0.0F;
    float first_denoised_mix = 0.0F;
    float second_denoised_mix = 0.0F;
};

[[nodiscard]] DPMppSDEStage make_dpmpp_sde_stage(float sigma_from,
                                                  float sigma_to,
                                                  float eta,
                                                  float r);

struct BrownianIntervalWeight {
    std::size_t interval_index = 0;
    float coefficient = 0.0F;
};

class BrownianIntervalPlan final {
public:
    explicit BrownianIntervalPlan(std::vector<float> points);

    [[nodiscard]] const std::vector<float>& points() const noexcept { return points_; }
    [[nodiscard]] std::size_t interval_count() const noexcept {
        return points_.empty() ? 0 : points_.size() - 1;
    }
    [[nodiscard]] std::vector<BrownianIntervalWeight> weights(float from,
                                                               float to) const;

private:
    [[nodiscard]] std::size_t point_index(float value) const;
    std::vector<float> points_;
};

[[nodiscard]] std::uint64_t brownian_interval_seed(std::uint64_t seed,
                                                    std::size_t interval_index) noexcept;

struct SamplerConfig {
    float eta = 1.0F;
    float s_noise = 1.0F;
    float r = 0.5F;
    float s_churn = 0.0F;
    float s_tmin = 0.0F;
    float s_tmax = 1.0e30F;
    NoiseDevice noise_device = NoiseDevice::CPU;
    SamplerStatePrecision state_precision = SamplerStatePrecision::Float32;
    InitialNoiseScaling initial_noise_scaling = InitialNoiseScaling::ComfyUIMaxDenoise;
};

struct SchedulerConfig {
    std::size_t training_timesteps = 1000;
    float beta_start = 0.00085F;
    float beta_end = 0.012F;
    PredictionType prediction_type = PredictionType::Epsilon;
    float karras_rho = 7.0F;
    float beta_schedule_alpha = 0.6F;
    float beta_schedule_beta = 0.6F;
    float linear_quadratic_threshold = 0.025F;
    float gits_coeff = 1.20F;
    float denoise = 1.0F;
    bool set_alpha_to_one = false;
};

class SigmaSchedule final {
public:
    explicit SigmaSchedule(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps,
                       SchedulerKind scheduler = SchedulerKind::Normal);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    [[nodiscard]] const std::vector<float>& sigmas() const noexcept { return sigmas_; }
    [[nodiscard]] const std::vector<float>& training_sigmas() const noexcept {
        return training_sigmas_;
    }
    [[nodiscard]] float initial_noise_sigma() const;
    [[nodiscard]] float final_alpha_cumprod() const noexcept;
    [[nodiscard]] SchedulerKind scheduler_kind() const noexcept { return scheduler_kind_; }
    [[nodiscard]] float timestep_for(float sigma) const;

private:
    SchedulerConfig config_;
    SchedulerKind scheduler_kind_ = SchedulerKind::Normal;
    std::vector<float> training_sigmas_;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

class EulerDiscreteScheduler final {
public:
    explicit EulerDiscreteScheduler(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps,
                       SchedulerKind scheduler = SchedulerKind::Normal);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept {
        return schedule_.timesteps();
    }
    [[nodiscard]] const std::vector<float>& sigmas() const noexcept {
        return schedule_.sigmas();
    }
    [[nodiscard]] float initial_noise_sigma() const { return schedule_.initial_noise_sigma(); }

    [[nodiscard]] FloatTensor scale_model_input(const FloatTensor& sample,
                                                std::size_t step_index) const;
    [[nodiscard]] FloatTensor step(const FloatTensor& model_output,
                                   std::size_t step_index,
                                   const FloatTensor& sample) const;

private:
    SchedulerConfig config_;
    SigmaSchedule schedule_;
};

class DPMpp2MSampler final {
public:
    explicit DPMpp2MSampler(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps,
                       SchedulerKind scheduler = SchedulerKind::Normal);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept {
        return schedule_.timesteps();
    }
    [[nodiscard]] const std::vector<float>& sigmas() const noexcept {
        return schedule_.sigmas();
    }
    [[nodiscard]] float initial_noise_sigma() const { return schedule_.initial_noise_sigma(); }

    [[nodiscard]] FloatTensor scale_model_input(const FloatTensor& sample,
                                                std::size_t step_index) const;
    [[nodiscard]] FloatTensor step(const FloatTensor& model_output,
                                   std::size_t step_index,
                                   const FloatTensor& sample);
    void reset() noexcept { old_denoised_.reset(); }

private:
    SchedulerConfig config_;
    SigmaSchedule schedule_;
    std::optional<FloatTensor> old_denoised_;
};

class DDIMScheduler final {
public:
    explicit DDIMScheduler(SchedulerConfig config = {});

    void set_timesteps(std::size_t inference_steps,
                       SchedulerKind scheduler = SchedulerKind::Normal);

    [[nodiscard]] const std::vector<float>& timesteps() const noexcept {
        return schedule_.timesteps();
    }
    [[nodiscard]] const std::vector<float>& sigmas() const noexcept {
        return schedule_.sigmas();
    }
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
    SigmaSchedule schedule_;
};

[[nodiscard]] FloatTensor random_normal_tensor(const std::vector<std::size_t>& shape,
                                               std::uint64_t seed);

} // namespace sdxl
