#include "sdxl/scheduler.hpp"

#include "sdxl/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>

namespace sdxl {
namespace {

[[nodiscard]] std::size_t checked_element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) throw Error("tensor shape cannot be empty");
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) throw Error("tensor dimensions must be nonzero");
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw Error("tensor element count overflows size_t");
        }
        count *= dimension;
    }
    return count;
}

void require_same_shape(const FloatTensor& first,
                        const FloatTensor& second,
                        const char* operation) {
    if (first.shape != second.shape || first.values.size() != second.values.size()) {
        throw Error(std::string(operation) + " requires tensors with equal shapes");
    }
}

[[nodiscard]] std::vector<float> make_scaled_linear_betas(const SchedulerConfig& config) {
    if (config.training_timesteps < 2) throw Error("scheduler requires at least two training timesteps");
    if (!(config.beta_start > 0.0F && config.beta_end > config.beta_start && config.beta_end < 1.0F)) {
        throw Error("scheduler beta range is invalid");
    }
    std::vector<float> betas(config.training_timesteps);
    const double start = std::sqrt(static_cast<double>(config.beta_start));
    const double end = std::sqrt(static_cast<double>(config.beta_end));
    const double denominator = static_cast<double>(config.training_timesteps - 1);
    for (std::size_t index = 0; index < config.training_timesteps; ++index) {
        const double fraction = static_cast<double>(index) / denominator;
        const double value = start + (end - start) * fraction;
        betas[index] = static_cast<float>(value * value);
    }
    return betas;
}

[[nodiscard]] std::vector<float> make_alphas_cumprod(const SchedulerConfig& config) {
    const std::vector<float> betas = make_scaled_linear_betas(config);
    std::vector<float> cumulative(betas.size());
    double product = 1.0;
    for (std::size_t index = 0; index < betas.size(); ++index) {
        product *= 1.0 - static_cast<double>(betas[index]);
        cumulative[index] = static_cast<float>(product);
    }
    return cumulative;
}

[[nodiscard]] float interpolate(const std::vector<float>& values, float position) {
    if (values.empty()) throw Error("cannot interpolate an empty table");
    const float clamped = std::clamp(position, 0.0F, static_cast<float>(values.size() - 1));
    const auto lower = static_cast<std::size_t>(std::floor(clamped));
    const auto upper = std::min(values.size() - 1, lower + 1);
    const float fraction = clamped - static_cast<float>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

[[nodiscard]] double unit_uniform(std::mt19937_64& generator) noexcept {
    // 53 random bits mapped into the open interval (0, 1).
    constexpr double inverse = 1.0 / 9007199254740992.0;
    const std::uint64_t bits = generator() >> 11U;
    return (static_cast<double>(bits) + 0.5) * inverse;
}

void fill_normal(std::vector<float>& destination, std::mt19937_64& generator) {
    std::size_t index = 0;
    while (index < destination.size()) {
        const double u1 = unit_uniform(generator);
        const double u2 = unit_uniform(generator);
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * std::numbers::pi * u2;
        destination[index++] = static_cast<float>(radius * std::cos(angle));
        if (index < destination.size()) {
            destination[index++] = static_cast<float>(radius * std::sin(angle));
        }
    }
}

} // namespace

EulerDiscreteScheduler::EulerDiscreteScheduler(SchedulerConfig config)
    : config_(config) {
    const std::vector<float> cumulative = make_alphas_cumprod(config_);
    training_sigmas_.resize(cumulative.size());
    for (std::size_t index = 0; index < cumulative.size(); ++index) {
        const double alpha = static_cast<double>(cumulative[index]);
        training_sigmas_[index] = static_cast<float>(std::sqrt((1.0 - alpha) / alpha));
    }
}

void EulerDiscreteScheduler::set_timesteps(std::size_t inference_steps) {
    if (inference_steps == 0) throw Error("inference step count must be positive");
    timesteps_.resize(inference_steps);
    sigmas_.resize(inference_steps + 1);
    if (inference_steps == 1) {
        // Matches linspace(0, training_timesteps - 1, 1) followed by reversal.
        timesteps_[0] = 0.0F;
        sigmas_[0] = training_sigmas_.front();
    } else {
        const float maximum = static_cast<float>(config_.training_timesteps - 1);
        const float denominator = static_cast<float>(inference_steps - 1);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const float timestep = maximum - maximum * static_cast<float>(index) / denominator;
            timesteps_[index] = timestep;
            sigmas_[index] = interpolate(training_sigmas_, timestep);
        }
    }
    sigmas_.back() = 0.0F;
}

float EulerDiscreteScheduler::initial_noise_sigma() const {
    if (sigmas_.empty()) throw Error("Euler scheduler timesteps have not been initialized");
    return *std::max_element(sigmas_.begin(), sigmas_.end());
}

FloatTensor EulerDiscreteScheduler::scale_model_input(const FloatTensor& sample,
                                                       std::size_t step_index) const {
    if (step_index >= timesteps_.size() || step_index >= sigmas_.size()) {
        throw Error("Euler scheduler step index is out of range");
    }
    const float sigma = sigmas_[step_index];
    const float scale = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    FloatTensor result{sample.shape, sample.values};
    for (float& value : result.values) value *= scale;
    return result;
}

FloatTensor EulerDiscreteScheduler::step(const FloatTensor& model_output,
                                          std::size_t step_index,
                                          const FloatTensor& sample) const {
    require_same_shape(model_output, sample, "Euler scheduler step");
    if (step_index >= timesteps_.size() || step_index + 1 >= sigmas_.size()) {
        throw Error("Euler scheduler step index is out of range");
    }
    const float sigma = sigmas_[step_index];
    const float sigma_next = sigmas_[step_index + 1];
    const float dt = sigma_next - sigma;
    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};

    for (std::size_t index = 0; index < sample.values.size(); ++index) {
        const float current = sample.values[index];
        const float prediction = model_output.values[index];
        float predicted_original = 0.0F;
        switch (config_.prediction_type) {
        case PredictionType::Epsilon:
            predicted_original = current - sigma * prediction;
            break;
        case PredictionType::Sample:
            predicted_original = prediction;
            break;
        case PredictionType::VPrediction:
            predicted_original = prediction * (-sigma / std::sqrt(sigma * sigma + 1.0F)) +
                                 current / (sigma * sigma + 1.0F);
            break;
        }
        const float derivative = sigma > 0.0F ? (current - predicted_original) / sigma : 0.0F;
        result.values[index] = current + derivative * dt;
    }
    return result;
}

DDIMScheduler::DDIMScheduler(SchedulerConfig config)
    : config_(config), alphas_cumprod_(make_alphas_cumprod(config_)) {}

void DDIMScheduler::set_timesteps(std::size_t inference_steps) {
    if (inference_steps == 0) throw Error("inference step count must be positive");
    if (inference_steps > config_.training_timesteps) {
        throw Error("DDIM inference steps cannot exceed training timesteps");
    }
    const std::size_t step_ratio = config_.training_timesteps / inference_steps;
    if (step_ratio == 0) throw Error("DDIM timestep ratio is zero");
    timesteps_.resize(inference_steps);
    for (std::size_t index = 0; index < inference_steps; ++index) {
        timesteps_[index] = static_cast<float>((inference_steps - 1 - index) * step_ratio);
    }
}

FloatTensor DDIMScheduler::scale_model_input(const FloatTensor& sample,
                                              std::size_t step_index) const {
    if (step_index >= timesteps_.size()) throw Error("DDIM scheduler step index is out of range");
    return sample;
}

FloatTensor DDIMScheduler::step(const FloatTensor& model_output,
                                 std::size_t step_index,
                                 const FloatTensor& sample,
                                 float eta,
                                 std::mt19937_64* generator) const {
    require_same_shape(model_output, sample, "DDIM scheduler step");
    if (step_index >= timesteps_.size()) throw Error("DDIM scheduler step index is out of range");
    if (eta < 0.0F) throw Error("DDIM eta cannot be negative");

    const auto timestep = static_cast<std::size_t>(std::llround(timesteps_[step_index]));
    if (timestep >= alphas_cumprod_.size()) throw Error("DDIM timestep exceeds training range");
    const float alpha_t = alphas_cumprod_[timestep];
    const float alpha_previous = step_index + 1 < timesteps_.size()
                                     ? alphas_cumprod_[static_cast<std::size_t>(
                                           std::llround(timesteps_[step_index + 1]))]
                                     : 1.0F;
    const float beta_t = 1.0F - alpha_t;
    const float sqrt_alpha_t = std::sqrt(alpha_t);
    const float sqrt_beta_t = std::sqrt(beta_t);

    const float variance = std::max(
        0.0F,
        ((1.0F - alpha_previous) / (1.0F - alpha_t)) *
            (1.0F - alpha_t / alpha_previous));
    const float standard_deviation = eta * std::sqrt(variance);
    const float direction_scale = std::sqrt(
        std::max(0.0F, 1.0F - alpha_previous - standard_deviation * standard_deviation));

    std::vector<float> noise;
    if (standard_deviation > 0.0F) {
        if (generator == nullptr) throw Error("DDIM eta > 0 requires a random generator");
        noise.resize(sample.values.size());
        fill_normal(noise, *generator);
    }

    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};
    for (std::size_t index = 0; index < sample.values.size(); ++index) {
        const float current = sample.values[index];
        const float prediction = model_output.values[index];
        float predicted_original = 0.0F;
        float predicted_epsilon = 0.0F;
        switch (config_.prediction_type) {
        case PredictionType::Epsilon:
            predicted_original = (current - sqrt_beta_t * prediction) / sqrt_alpha_t;
            predicted_epsilon = prediction;
            break;
        case PredictionType::Sample:
            predicted_original = prediction;
            predicted_epsilon = (current - sqrt_alpha_t * predicted_original) / sqrt_beta_t;
            break;
        case PredictionType::VPrediction:
            predicted_original = sqrt_alpha_t * current - sqrt_beta_t * prediction;
            predicted_epsilon = sqrt_alpha_t * prediction + sqrt_beta_t * current;
            break;
        }
        float previous = std::sqrt(alpha_previous) * predicted_original +
                         direction_scale * predicted_epsilon;
        if (!noise.empty()) previous += standard_deviation * noise[index];
        result.values[index] = previous;
    }
    return result;
}

FloatTensor random_normal_tensor(const std::vector<std::size_t>& shape,
                                 std::uint64_t seed) {
    FloatTensor result;
    result.shape = shape;
    result.values.resize(checked_element_count(shape));
    std::mt19937_64 generator(seed);
    fill_normal(result.values, generator);
    return result;
}

} // namespace sdxl
