#include "sdxl/scheduler.hpp"
#include "sdxl/gits_schedule_data.hpp"

#include "sdxl/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <iterator>
#include <numbers>
#include <string>
#include <utility>

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

[[nodiscard]] std::string normalized_name(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        if (c == '-') return '_';
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

[[nodiscard]] std::vector<float> make_scaled_linear_betas(const SchedulerConfig& config) {
    if (config.training_timesteps < 2) {
        throw Error("scheduler requires at least two training timesteps");
    }
    if (!(config.beta_start > 0.0F && config.beta_end > config.beta_start &&
          config.beta_end < 1.0F)) {
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

[[nodiscard]] std::vector<float> make_training_sigmas(const SchedulerConfig& config) {
    const std::vector<float> cumulative = make_alphas_cumprod(config);
    std::vector<float> sigmas(cumulative.size());
    for (std::size_t index = 0; index < cumulative.size(); ++index) {
        const double alpha = static_cast<double>(cumulative[index]);
        sigmas[index] = static_cast<float>(std::sqrt((1.0 - alpha) / alpha));
    }
    return sigmas;
}

[[nodiscard]] float sigma_for_timestep(const std::vector<float>& values,
                                         float position) {
    if (values.empty()) throw Error("cannot interpolate an empty sigma table");
    const float clamped = std::clamp(position, 0.0F,
                                     static_cast<float>(values.size() - 1));
    const auto lower = static_cast<std::size_t>(std::floor(clamped));
    const auto upper = std::min(values.size() - 1, lower + 1);
    const double fraction = static_cast<double>(clamped - static_cast<float>(lower));
    const double lower_log = std::log(static_cast<double>(values[lower]));
    const double upper_log = std::log(static_cast<double>(values[upper]));
    return static_cast<float>(std::exp(
        (1.0 - fraction) * lower_log + fraction * upper_log));
}

[[nodiscard]] float timestep_for_sigma(const std::vector<float>& training_sigmas,
                                       float sigma) {
    if (training_sigmas.empty()) throw Error("training sigma table is empty");
    if (!(sigma > 0.0F) || !std::isfinite(sigma)) {
        throw Error("cannot map a non-positive or non-finite sigma to a timestep");
    }
    if (sigma <= training_sigmas.front()) return 0.0F;
    if (sigma >= training_sigmas.back()) {
        return static_cast<float>(training_sigmas.size() - 1);
    }

    // ComfyUI ModelSamplingDiscrete::timestep performs an argmin in log-sigma
    // space and supplies that integer training timestep to the UNet. Do not
    // feed fractional inverse-interpolation timesteps here: that changes the
    // timestep embedding for generated schedules and DPM++ SDE midpoints.
    const auto upper_iterator = std::lower_bound(
        training_sigmas.begin(), training_sigmas.end(), sigma);
    const std::size_t upper = static_cast<std::size_t>(
        upper_iterator - training_sigmas.begin());
    const std::size_t lower = upper - 1;
    const double target_log = std::log(static_cast<double>(sigma));
    const double lower_distance = std::abs(
        target_log - std::log(static_cast<double>(training_sigmas[lower])));
    const double upper_distance = std::abs(
        std::log(static_cast<double>(training_sigmas[upper])) - target_log);
    return static_cast<float>(lower_distance <= upper_distance ? lower : upper);
}

[[nodiscard]] double beta_continued_fraction(double a, double b, double x) {
    constexpr int maximum_iterations = 200;
    constexpr double epsilon = 3.0e-14;
    constexpr double minimum = 1.0e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < minimum) d = minimum;
    d = 1.0 / d;
    double result = d;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        const int doubled = iteration * 2;
        double coefficient = static_cast<double>(iteration) *
                             (b - static_cast<double>(iteration)) * x /
                             ((qam + static_cast<double>(doubled)) *
                              (a + static_cast<double>(doubled)));
        d = 1.0 + coefficient * d;
        if (std::abs(d) < minimum) d = minimum;
        c = 1.0 + coefficient / c;
        if (std::abs(c) < minimum) c = minimum;
        d = 1.0 / d;
        result *= d * c;

        coefficient = -(a + static_cast<double>(iteration)) *
                      (qab + static_cast<double>(iteration)) * x /
                      ((a + static_cast<double>(doubled)) *
                       (qap + static_cast<double>(doubled)));
        d = 1.0 + coefficient * d;
        if (std::abs(d) < minimum) d = minimum;
        c = 1.0 + coefficient / c;
        if (std::abs(c) < minimum) c = minimum;
        d = 1.0 / d;
        const double delta = d * c;
        result *= delta;
        if (std::abs(delta - 1.0) <= epsilon) break;
    }
    return result;
}

[[nodiscard]] double regularized_incomplete_beta(double a, double b, double x) {
    if (!(a > 0.0 && b > 0.0)) throw Error("beta scheduler parameters must be positive");
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double factor = std::exp(
        std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
        a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return factor * beta_continued_fraction(a, b, x) / a;
    }
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b;
}

[[nodiscard]] double inverse_regularized_beta(double probability,
                                              double a,
                                              double b) {
    if (probability <= 0.0) return 0.0;
    if (probability >= 1.0) return 1.0;
    double lower = 0.0;
    double upper = 1.0;
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double middle = (lower + upper) * 0.5;
        if (regularized_incomplete_beta(a, b, middle) < probability) lower = middle;
        else upper = middle;
    }
    return (lower + upper) * 0.5;
}

void validate_sigma_schedule(const std::vector<float>& timesteps,
                             const std::vector<float>& sigmas) {
    if (timesteps.empty() || sigmas.size() != timesteps.size() + 1) {
        throw Error("sigma scheduler produced inconsistent lengths");
    }
    if (sigmas.back() != 0.0F) throw Error("sigma scheduler must terminate at zero");
    for (std::size_t index = 0; index < timesteps.size(); ++index) {
        if (!std::isfinite(timesteps[index]) || !std::isfinite(sigmas[index]) ||
            !(sigmas[index] > 0.0F)) {
            throw Error("sigma scheduler produced a non-finite or non-positive step");
        }
        if (index > 0 && !(sigmas[index] < sigmas[index - 1])) {
            throw Error("sigma scheduler must be strictly decreasing before zero");
        }
    }
}

[[nodiscard]] FloatTensor predicted_original_sample(const FloatTensor& model_output,
                                                    const FloatTensor& sample,
                                                    float sigma,
                                                    PredictionType prediction_type) {
    require_same_shape(model_output, sample, "predicted-original conversion");
    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};
    const float sigma_squared = sigma * sigma;
    const float normalization = std::sqrt(sigma_squared + 1.0F);
    for (std::size_t index = 0; index < sample.values.size(); ++index) {
        const float current = sample.values[index];
        const float prediction = model_output.values[index];
        switch (prediction_type) {
        case PredictionType::Epsilon:
            result.values[index] = current - sigma * prediction;
            break;
        case PredictionType::Sample:
            result.values[index] = prediction;
            break;
        case PredictionType::VPrediction:
            result.values[index] = prediction * (-sigma / normalization) +
                                   current / (sigma_squared + 1.0F);
            break;
        }
    }
    return result;
}

[[nodiscard]] float pytorch_uniform_float(std::mt19937& generator) noexcept {
    constexpr std::uint32_t mask = (1U << 24U) - 1U;
    constexpr float divisor = 1.0F / static_cast<float>(1U << 24U);
    return static_cast<float>(generator() & mask) * divisor;
}

[[nodiscard]] double pytorch_uniform_double(std::mt19937& generator) noexcept {
    constexpr std::uint64_t mask = (1ULL << 53U) - 1ULL;
    constexpr double divisor = 1.0 / static_cast<double>(1ULL << 53U);
    const std::uint64_t value =
        (static_cast<std::uint64_t>(generator()) << 32U) |
        static_cast<std::uint64_t>(generator());
    return static_cast<double>(value & mask) * divisor;
}

void pytorch_normal_fill16(float* data) {
    // This is the scalar form of PyTorch's contiguous Float32 CPU normal
    // kernel: eight Box-Muller pairs are formed from a 16-value uniform
    // block, with the second eight values used as the angles.
    for (std::size_t index = 0; index < 8; ++index) {
        const float uniform_radius = 1.0F - data[index];
        const float uniform_angle = data[index + 8];
        const float radius = std::sqrt(-2.0F * std::log(uniform_radius));
        const float angle = 2.0F * static_cast<float>(std::numbers::pi) * uniform_angle;
        data[index] = std::fma(radius * std::cos(angle), 1.0F, 0.0F);
        data[index + 8] = std::fma(radius * std::sin(angle), 1.0F, 0.0F);
    }
}

void fill_pytorch_cpu_normal(std::vector<float>& destination,
                             std::mt19937& generator) {
    if (destination.size() >= 16) {
        for (float& value : destination) value = pytorch_uniform_float(generator);
        for (std::size_t offset = 0; offset + 15 < destination.size(); offset += 16) {
            pytorch_normal_fill16(destination.data() + offset);
        }
        // PyTorch regenerates and overwrites the final 16 values when the
        // contiguous tensor length is not a multiple of 16.
        if (destination.size() % 16 != 0) {
            const std::size_t offset = destination.size() - 16;
            for (std::size_t index = 0; index < 16; ++index) {
                destination[offset + index] = pytorch_uniform_float(generator);
            }
            pytorch_normal_fill16(destination.data() + offset);
        }
        return;
    }

    // PyTorch's small/non-contiguous CPU path uses the double-precision
    // normal_distribution and caches the second Box-Muller result.
    bool cached = false;
    double cached_value = 0.0;
    for (float& value : destination) {
        if (cached) {
            value = static_cast<float>(cached_value);
            cached = false;
            continue;
        }
        const double uniform_angle = pytorch_uniform_double(generator);
        const double uniform_radius = pytorch_uniform_double(generator);
        const double radius = std::sqrt(-2.0 * std::log1p(-uniform_radius));
        const double angle = 2.0 * std::numbers::pi * uniform_angle;
        value = static_cast<float>(radius * std::cos(angle));
        cached_value = radius * std::sin(angle);
        cached = true;
    }
}

[[nodiscard]] double legacy_unit_uniform(std::mt19937_64& generator) noexcept {
    constexpr double inverse = 1.0 / 9007199254740992.0;
    const std::uint64_t bits = generator() >> 11U;
    return (static_cast<double>(bits) + 0.5) * inverse;
}

void fill_legacy_normal(std::vector<float>& destination,
                        std::mt19937_64& generator) {
    std::size_t index = 0;
    while (index < destination.size()) {
        const double u1 = legacy_unit_uniform(generator);
        const double u2 = legacy_unit_uniform(generator);
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * std::numbers::pi * u2;
        destination[index++] = static_cast<float>(radius * std::cos(angle));
        if (index < destination.size()) {
            destination[index++] = static_cast<float>(radius * std::sin(angle));
        }
    }
}

} // namespace

SamplerKind parse_sampler_kind(std::string value) {
    value = normalized_name(std::move(value));
    if (value == "dpmpp_2m" || value == "dpmpp2m" || value == "dpm++_2m" ||
        value == "dpm++2m") {
        return SamplerKind::DPMpp2M;
    }
    if (value == "dpmpp_sde" || value == "dpm++_sde") return SamplerKind::DPMppSDE;
    if (value == "euler" || value == "euler_discrete") return SamplerKind::Euler;
    if (value == "euler_ancestral" || value == "euler_a" || value == "samplereulerancestral") return SamplerKind::EulerAncestral;
    if (value == "dpmpp_2s_ancestral_cfg_pp" || value == "dpmpp_2s_ancestral_cfg++" ||
        value == "dpm++_2s_ancestral_cfg_pp") return SamplerKind::DPMpp2SAncestralCFGpp;
    if (value == "ddim") return SamplerKind::DDIM;
    throw Error("unsupported sampler: " + value);
}

SchedulerKind parse_scheduler_kind(std::string value) {
    value = normalized_name(std::move(value));
    if (value == "normal") return SchedulerKind::Normal;
    if (value == "karras") return SchedulerKind::Karras;
    if (value == "exponential") return SchedulerKind::Exponential;
    if (value == "sgm_uniform" || value == "sgm") return SchedulerKind::SGMUniform;
    if (value == "simple") return SchedulerKind::Simple;
    if (value == "ddim_uniform") return SchedulerKind::DDIMUniform;
    if (value == "ddim_trailing" || value == "trailing" ||
        value == "hyper_sdxl" || value == "hyper_sdxl_ddim") {
        return SchedulerKind::DDIMTrailing;
    }
    if (value == "beta") return SchedulerKind::Beta;
    if (value == "linear_quadratic") return SchedulerKind::LinearQuadratic;
    if (value == "kl_optimal") return SchedulerKind::KLOptimal;
    if (value == "gits" || value == "gits_scheduler" || value == "gitsscheduler") return SchedulerKind::GITS;
    throw Error("unsupported scheduler: " + value);
}

const char* sampler_kind_name(SamplerKind kind) noexcept {
    switch (kind) {
    case SamplerKind::DPMpp2M: return "dpmpp_2m";
    case SamplerKind::DPMppSDE: return "dpmpp_sde";
    case SamplerKind::Euler: return "euler";
    case SamplerKind::EulerAncestral: return "euler_ancestral";
    case SamplerKind::DPMpp2SAncestralCFGpp: return "dpmpp_2s_ancestral_cfg_pp";
    case SamplerKind::DDIM: return "ddim";
    }
    return "unknown";
}

const char* scheduler_kind_name(SchedulerKind kind) noexcept {
    switch (kind) {
    case SchedulerKind::Normal: return "normal";
    case SchedulerKind::Karras: return "karras";
    case SchedulerKind::Exponential: return "exponential";
    case SchedulerKind::SGMUniform: return "sgm_uniform";
    case SchedulerKind::Simple: return "simple";
    case SchedulerKind::DDIMUniform: return "ddim_uniform";
    case SchedulerKind::DDIMTrailing: return "ddim_trailing";
    case SchedulerKind::Beta: return "beta";
    case SchedulerKind::LinearQuadratic: return "linear_quadratic";
    case SchedulerKind::KLOptimal: return "kl_optimal";
    case SchedulerKind::GITS: return "gits";
    }
    return "unknown";
}

bool is_sampler_kind_name(std::string_view value) noexcept {
    try {
        (void)parse_sampler_kind(std::string(value));
        return true;
    } catch (...) {
        return false;
    }
}

bool is_scheduler_kind_name(std::string_view value) noexcept {
    try {
        (void)parse_scheduler_kind(std::string(value));
        return true;
    } catch (...) {
        return false;
    }
}


NoiseDevice parse_noise_device(std::string value) {
    value = normalized_name(std::move(value));
    if (value == "cpu") return NoiseDevice::CPU;
    if (value == "gpu" || value == "cuda") return NoiseDevice::GPU;
    throw Error("unsupported noise device: " + value);
}

const char* noise_device_name(NoiseDevice device) noexcept {
    return device == NoiseDevice::CPU ? "cpu" : "gpu";
}

SamplerStatePrecision parse_sampler_state_precision(std::string value) {
    value = normalized_name(std::move(value));
    if (value == "fp16" || value == "float16" || value == "half") {
        return SamplerStatePrecision::Float16;
    }
    if (value == "fp32" || value == "float32" || value == "comfyui") {
        return SamplerStatePrecision::Float32;
    }
    throw Error("unsupported sampler-state precision: " + value);
}

const char* sampler_state_precision_name(SamplerStatePrecision value) noexcept {
    return value == SamplerStatePrecision::Float32 ? "fp32" : "fp16";
}

InitialNoiseScaling parse_initial_noise_scaling(std::string value) {
    value = normalized_name(std::move(value));
    if (value == "sigma" || value == "legacy") return InitialNoiseScaling::Sigma;
    if (value == "comfyui" || value == "max_denoise" ||
        value == "comfyui_max_denoise") {
        return InitialNoiseScaling::ComfyUIMaxDenoise;
    }
    throw Error("unsupported initial-noise scaling: " + value);
}

const char* initial_noise_scaling_name(InitialNoiseScaling value) noexcept {
    return value == InitialNoiseScaling::ComfyUIMaxDenoise
        ? "comfyui-max-denoise" : "sigma";
}

AncestralStep make_ancestral_step(float sigma_from, float sigma_to, float eta) {
    if (!(sigma_from > 0.0F) || sigma_to < 0.0F || sigma_to > sigma_from ||
        !std::isfinite(sigma_from) || !std::isfinite(sigma_to)) {
        throw Error("ancestral step requires finite decreasing nonnegative sigmas");
    }
    if (!std::isfinite(eta) || eta < 0.0F) {
        throw Error("sampler eta must be finite and nonnegative");
    }
    if (eta == 0.0F || sigma_to == 0.0F) return {sigma_to, 0.0F};

    const double from_squared = static_cast<double>(sigma_from) * sigma_from;
    const double to_squared = static_cast<double>(sigma_to) * sigma_to;
    const double variance = std::max(
        0.0, to_squared * (from_squared - to_squared) / from_squared);
    const double sigma_up = std::min(
        static_cast<double>(sigma_to),
        static_cast<double>(eta) * std::sqrt(variance));
    const double sigma_down = std::sqrt(std::max(0.0, to_squared - sigma_up * sigma_up));
    return {static_cast<float>(sigma_down), static_cast<float>(sigma_up)};
}

DPMppSDEStage make_dpmpp_sde_stage(float sigma_from,
                                    float sigma_to,
                                    float eta,
                                    float r) {
    if (!(sigma_from > sigma_to) || !(sigma_to > 0.0F) ||
        !std::isfinite(sigma_from) || !std::isfinite(sigma_to)) {
        throw Error("DPM++ SDE stage requires finite positive decreasing sigmas");
    }
    if (!std::isfinite(r) || !(r > 0.0F) || r > 1.0F) {
        throw Error("DPM++ SDE midpoint ratio must be in (0, 1]");
    }

    const double lambda_from = -std::log(static_cast<double>(sigma_from));
    const double lambda_to = -std::log(static_cast<double>(sigma_to));
    const double lambda_mid = lambda_from +
        static_cast<double>(r) * (lambda_to - lambda_from);

    DPMppSDEStage stage;
    stage.sigma_mid = static_cast<float>(std::exp(-lambda_mid));
    if (!(stage.sigma_mid < sigma_from) || stage.sigma_mid < sigma_to) {
        throw Error("DPM++ SDE midpoint ratio collapses at float precision");
    }
    stage.midpoint = make_ancestral_step(sigma_from, stage.sigma_mid, eta);
    stage.final = make_ancestral_step(sigma_from, sigma_to, eta);

    // For the standard SD/SDXL epsilon parameterization used here,
    // alpha(lambda) is one in K-diffusion's external sigma space. The
    // ComfyUI/k-diffusion DPM++ SDE equations therefore reduce exactly to
    // these sigma-down ratios. This form is also well-defined when a large
    // eta drives sigma_down to zero.
    stage.midpoint_sample_coefficient = stage.midpoint.down / sigma_from;
    stage.midpoint_denoised_coefficient =
        1.0F - stage.midpoint_sample_coefficient;
    stage.final_sample_coefficient = stage.final.down / sigma_from;
    stage.final_denoised_coefficient = 1.0F - stage.final_sample_coefficient;

    const float fac = 1.0F / (2.0F * r);
    stage.first_denoised_mix = 1.0F - fac;
    stage.second_denoised_mix = fac;
    return stage;
}

BrownianIntervalPlan::BrownianIntervalPlan(std::vector<float> points)
    : points_(std::move(points)) {
    if (points_.size() < 2) {
        throw Error("Brownian interval plan requires at least two points");
    }
    for (const float value : points_) {
        if (!(value > 0.0F) || !std::isfinite(value)) {
            throw Error("Brownian interval points must be finite and positive");
        }
    }
    std::sort(points_.begin(), points_.end());
    points_.erase(std::unique(points_.begin(), points_.end()), points_.end());
    if (points_.size() < 2) {
        throw Error("Brownian interval plan collapsed to fewer than two unique points");
    }
}

std::size_t BrownianIntervalPlan::point_index(float value) const {
    if (!(value > 0.0F) || !std::isfinite(value)) {
        throw Error("Brownian interval endpoint must be finite and positive");
    }
    const auto iterator = std::lower_bound(points_.begin(), points_.end(), value);
    auto matches = [&](std::vector<float>::const_iterator candidate) {
        if (candidate == points_.end()) return false;
        const float scale = std::max({1.0F, std::abs(value), std::abs(*candidate)});
        return std::abs(*candidate - value) <= 1.0e-6F * scale;
    };
    if (matches(iterator)) return static_cast<std::size_t>(iterator - points_.begin());
    if (iterator != points_.begin()) {
        const auto previous = std::prev(iterator);
        if (matches(previous)) return static_cast<std::size_t>(previous - points_.begin());
    }
    throw Error("Brownian interval endpoint is not present in the plan");
}

std::vector<BrownianIntervalWeight> BrownianIntervalPlan::weights(float from,
                                                                  float to) const {
    const std::size_t from_index = point_index(from);
    const std::size_t to_index = point_index(to);
    if (from_index == to_index) {
        throw Error("Brownian interval endpoints must differ");
    }

    const std::size_t lower = std::min(from_index, to_index);
    const std::size_t upper = std::max(from_index, to_index);
    const double total = std::abs(static_cast<double>(to) - static_cast<double>(from));
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw Error("Brownian interval has invalid length");
    }
    const float sign = to_index > from_index ? 1.0F : -1.0F;
    std::vector<BrownianIntervalWeight> result;
    result.reserve(upper - lower);
    for (std::size_t interval = lower; interval < upper; ++interval) {
        const double width = static_cast<double>(points_[interval + 1]) -
                             static_cast<double>(points_[interval]);
        if (!(width > 0.0) || !std::isfinite(width)) {
            throw Error("Brownian interval plan contains an invalid segment");
        }
        result.push_back(BrownianIntervalWeight{
            interval,
            sign * static_cast<float>(std::sqrt(width / total))
        });
    }
    return result;
}

std::uint64_t brownian_interval_seed(std::uint64_t seed,
                                     std::size_t interval_index) noexcept {
    // SplitMix64 gives every elementary Brownian interval an independent,
    // deterministic stream while preserving the caller's per-image seed.
    std::uint64_t value = seed +
        0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(interval_index) + 1ULL);
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

SigmaSchedule::SigmaSchedule(SchedulerConfig config)
    : config_(config), training_sigmas_(make_training_sigmas(config_)) {}

void SigmaSchedule::set_timesteps(std::size_t inference_steps,
                                  SchedulerKind scheduler) {
    if (inference_steps == 0) throw Error("inference step count must be positive");
    if (!std::isfinite(config_.denoise) || config_.denoise <= 0.0F ||
        config_.denoise > 1.0F) {
        throw Error("scheduler denoise must be finite and in (0,1]");
    }
    scheduler_kind_ = scheduler;
    timesteps_.clear();
    sigmas_.clear();

    // Match ComfyUI KSampler.set_steps(): for partial denoise, construct the
    // longer full schedule with int(steps / denoise), then keep its final
    // steps + 1 sigmas. GITS owns its separate denoise interpretation.
    if (scheduler != SchedulerKind::GITS && config_.denoise < 0.9999F) {
        const std::size_t expanded_steps = std::max(
            inference_steps,
            static_cast<std::size_t>(static_cast<double>(inference_steps) /
                                     static_cast<double>(config_.denoise)));
        SchedulerConfig full_config = config_;
        full_config.denoise = 1.0F;
        SigmaSchedule full_schedule(full_config);
        full_schedule.set_timesteps(expanded_steps, scheduler);
        if (full_schedule.sigmas().size() < inference_steps + 1U ||
            full_schedule.timesteps().size() < inference_steps) {
            throw Error("partial-denoise schedule does not contain enough steps");
        }
        sigmas_.assign(full_schedule.sigmas().end() -
                           static_cast<std::ptrdiff_t>(inference_steps + 1U),
                       full_schedule.sigmas().end());
        timesteps_.assign(full_schedule.timesteps().end() -
                              static_cast<std::ptrdiff_t>(inference_steps),
                          full_schedule.timesteps().end());
        validate_sigma_schedule(timesteps_, sigmas_);
        return;
    }
    timesteps_.reserve(inference_steps);
    sigmas_.reserve(inference_steps + 1);

    const std::size_t training_count = training_sigmas_.size();
    const float maximum_timestep = static_cast<float>(training_count - 1);
    const float sigma_min = training_sigmas_.front();
    const float sigma_max = training_sigmas_.back();

    auto append_from_timestep = [&](float timestep) {
        const float sigma = sigma_for_timestep(training_sigmas_, timestep);
        sigmas_.push_back(sigma);
        timesteps_.push_back(timestep_for_sigma(training_sigmas_, sigma));
    };
    auto append_generated_sigma = [&](float sigma) {
        sigmas_.push_back(sigma);
        timesteps_.push_back(timestep_for_sigma(training_sigmas_, sigma));
    };

    switch (scheduler) {
    case SchedulerKind::Normal: {
        if (inference_steps == 1) {
            append_from_timestep(maximum_timestep);
        } else {
            const float denominator = static_cast<float>(inference_steps - 1);
            for (std::size_t index = 0; index < inference_steps; ++index) {
                const float timestep = maximum_timestep -
                    maximum_timestep * static_cast<float>(index) / denominator;
                append_from_timestep(timestep);
            }
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::SGMUniform: {
        const float denominator = static_cast<float>(inference_steps);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const float timestep = maximum_timestep -
                maximum_timestep * static_cast<float>(index) / denominator;
            append_from_timestep(timestep);
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::Simple: {
        const double stride = static_cast<double>(training_count) /
                              static_cast<double>(inference_steps);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const std::size_t offset = std::min(
                training_count - 1,
                static_cast<std::size_t>(std::floor(static_cast<double>(index) * stride)));
            const std::size_t training_index = training_count - 1 - offset;
            append_from_timestep(static_cast<float>(training_index));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::DDIMUniform: {
        if (inference_steps > training_count) {
            throw Error("DDIM-uniform steps cannot exceed training timesteps");
        }
        const std::size_t stride = std::max<std::size_t>(training_count / inference_steps, 1);
        std::vector<std::size_t> selected;
        for (std::size_t training_index = 1; training_index < training_count;
             training_index += stride) {
            selected.push_back(training_index);
        }
        for (auto iterator = selected.rbegin(); iterator != selected.rend(); ++iterator) {
            append_from_timestep(static_cast<float>(*iterator));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::DDIMTrailing: {
        if (inference_steps > training_count) {
            throw Error("DDIM-trailing steps cannot exceed training timesteps");
        }
        // Exact Hugging Face Diffusers DDIMScheduler trailing spacing:
        // round(arange(num_train_timesteps, 0, -N/steps)) - 1. Hyper-SDXL
        // fixed 2/4/8-step LoRAs were published for this timestep recipe.
        const double step_ratio = static_cast<double>(training_count) /
                                  static_cast<double>(inference_steps);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const double raw = static_cast<double>(training_count) -
                               static_cast<double>(index) * step_ratio;
            const long long rounded = std::llround(raw) - 1LL;
            const long long clamped = std::clamp<long long>(
                rounded, 0LL, static_cast<long long>(training_count - 1));
            append_from_timestep(static_cast<float>(clamped));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::Karras: {
        if (!(config_.karras_rho > 0.0F)) throw Error("Karras rho must be positive");
        const double inverse_rho = 1.0 / static_cast<double>(config_.karras_rho);
        const double maximum_root = std::pow(static_cast<double>(sigma_max), inverse_rho);
        const double minimum_root = std::pow(static_cast<double>(sigma_min), inverse_rho);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const double ramp = inference_steps == 1 ? 0.0 :
                static_cast<double>(index) / static_cast<double>(inference_steps - 1);
            const double root = maximum_root + ramp * (minimum_root - maximum_root);
            append_generated_sigma(static_cast<float>(std::pow(root, config_.karras_rho)));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::Exponential: {
        const double log_maximum = std::log(static_cast<double>(sigma_max));
        const double log_minimum = std::log(static_cast<double>(sigma_min));
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const double ramp = inference_steps == 1 ? 0.0 :
                static_cast<double>(index) / static_cast<double>(inference_steps - 1);
            append_generated_sigma(static_cast<float>(
                std::exp(log_maximum + ramp * (log_minimum - log_maximum))));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::Beta: {
        const double alpha = static_cast<double>(config_.beta_schedule_alpha);
        const double beta = static_cast<double>(config_.beta_schedule_beta);
        std::size_t previous_index = training_count;
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const double probability = 1.0 -
                static_cast<double>(index) / static_cast<double>(inference_steps);
            const double quantile = inverse_regularized_beta(probability, alpha, beta);
            const auto rounded = static_cast<std::size_t>(std::nearbyint(
                quantile * static_cast<double>(training_count - 1)));
            const std::size_t training_index = std::min(training_count - 1, rounded);
            if (training_index == previous_index) continue;
            previous_index = training_index;
            append_from_timestep(static_cast<float>(training_index));
        }
        if (timesteps_.empty()) append_from_timestep(maximum_timestep);
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::LinearQuadratic: {
        if (inference_steps == 1) {
            append_generated_sigma(sigma_max);
            sigmas_.push_back(0.0F);
            break;
        }
        const std::size_t linear_steps = std::max<std::size_t>(1, inference_steps / 2);
        const std::size_t quadratic_steps = inference_steps - linear_steps;
        const double threshold = static_cast<double>(config_.linear_quadratic_threshold);
        if (!(threshold > 0.0 && threshold < 1.0)) {
            throw Error("linear-quadratic threshold must be in (0,1)");
        }
        std::vector<double> noise_schedule;
        noise_schedule.reserve(inference_steps + 1);
        for (std::size_t index = 0; index < linear_steps; ++index) {
            noise_schedule.push_back(
                static_cast<double>(index) * threshold / static_cast<double>(linear_steps));
        }
        if (quadratic_steps > 0) {
            const double threshold_difference = static_cast<double>(linear_steps) -
                                                threshold * static_cast<double>(inference_steps);
            const double quadratic_denominator =
                static_cast<double>(linear_steps) *
                static_cast<double>(quadratic_steps) * static_cast<double>(quadratic_steps);
            const double quadratic_coefficient = threshold_difference / quadratic_denominator;
            const double linear_coefficient = threshold / static_cast<double>(linear_steps) -
                2.0 * threshold_difference /
                    (static_cast<double>(quadratic_steps) *
                     static_cast<double>(quadratic_steps));
            const double constant = quadratic_coefficient *
                                    static_cast<double>(linear_steps) *
                                    static_cast<double>(linear_steps);
            for (std::size_t index = linear_steps; index < inference_steps; ++index) {
                const double position = static_cast<double>(index);
                noise_schedule.push_back(quadratic_coefficient * position * position +
                                         linear_coefficient * position + constant);
            }
        }
        noise_schedule.push_back(1.0);
        for (std::size_t index = 0; index < inference_steps; ++index) {
            append_generated_sigma(static_cast<float>(
                (1.0 - noise_schedule[index]) * static_cast<double>(sigma_max)));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::KLOptimal: {
        const double minimum_angle = std::atan(static_cast<double>(sigma_min));
        const double maximum_angle = std::atan(static_cast<double>(sigma_max));
        for (std::size_t index = 0; index < inference_steps; ++index) {
            const double fraction = inference_steps == 1 ? 0.0 :
                static_cast<double>(index) / static_cast<double>(inference_steps - 1);
            const double angle = fraction * minimum_angle +
                                 (1.0 - fraction) * maximum_angle;
            append_generated_sigma(static_cast<float>(std::tan(angle)));
        }
        sigmas_.push_back(0.0F);
        break;
    }
    case SchedulerKind::GITS: {
        sigmas_ = make_gits_sigmas(inference_steps, config_.gits_coeff, config_.denoise);
        if (sigmas_.size() < 2) throw Error("GITS denoise produced no sampling steps");
        timesteps_.reserve(sigmas_.size() - 1);
        for (std::size_t index = 0; index + 1 < sigmas_.size(); ++index) {
            timesteps_.push_back(timestep_for_sigma(training_sigmas_, sigmas_[index]));
        }
        break;
    }
    }

    validate_sigma_schedule(timesteps_, sigmas_);
}

float SigmaSchedule::timestep_for(float sigma) const {
    return timestep_for_sigma(training_sigmas_, sigma);
}

float SigmaSchedule::initial_noise_sigma() const {
    if (sigmas_.empty()) throw Error("sigma scheduler timesteps have not been initialized");
    return sigmas_.front();
}

float SigmaSchedule::final_alpha_cumprod() const noexcept {
    if (config_.set_alpha_to_one) return 1.0F;
    const float sigma_zero = training_sigmas_.front();
    return 1.0F / (sigma_zero * sigma_zero + 1.0F);
}

EulerDiscreteScheduler::EulerDiscreteScheduler(SchedulerConfig config)
    : config_(config), schedule_(config) {}

void EulerDiscreteScheduler::set_timesteps(std::size_t inference_steps,
                                           SchedulerKind scheduler) {
    schedule_.set_timesteps(inference_steps, scheduler);
}

FloatTensor EulerDiscreteScheduler::scale_model_input(const FloatTensor& sample,
                                                       std::size_t step_index) const {
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("Euler sampler step index is out of range");
    }
    const float sigma = schedule_.sigmas()[step_index];
    const float scale = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    FloatTensor result{sample.shape, sample.values};
    for (float& value : result.values) value *= scale;
    return result;
}

FloatTensor EulerDiscreteScheduler::step(const FloatTensor& model_output,
                                         std::size_t step_index,
                                         const FloatTensor& sample) const {
    require_same_shape(model_output, sample, "Euler sampler step");
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("Euler sampler step index is out of range");
    }
    const float sigma = schedule_.sigmas()[step_index];
    const float sigma_next = schedule_.sigmas()[step_index + 1];
    const float delta = sigma_next - sigma;
    const FloatTensor denoised = predicted_original_sample(
        model_output, sample, sigma, config_.prediction_type);
    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};
    for (std::size_t index = 0; index < sample.values.size(); ++index) {
        const float derivative = sigma > 0.0F
            ? (sample.values[index] - denoised.values[index]) / sigma : 0.0F;
        result.values[index] = sample.values[index] + derivative * delta;
    }
    return result;
}

DPMpp2MSampler::DPMpp2MSampler(SchedulerConfig config)
    : config_(config), schedule_(config) {}

void DPMpp2MSampler::set_timesteps(std::size_t inference_steps,
                                   SchedulerKind scheduler) {
    schedule_.set_timesteps(inference_steps, scheduler);
    reset();
}

FloatTensor DPMpp2MSampler::scale_model_input(const FloatTensor& sample,
                                              std::size_t step_index) const {
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("DPM++ 2M sampler step index is out of range");
    }
    const float sigma = schedule_.sigmas()[step_index];
    const float scale = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    FloatTensor result{sample.shape, sample.values};
    for (float& value : result.values) value *= scale;
    return result;
}

FloatTensor DPMpp2MSampler::step(const FloatTensor& model_output,
                                 std::size_t step_index,
                                 const FloatTensor& sample) {
    require_same_shape(model_output, sample, "DPM++ 2M sampler step");
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("DPM++ 2M sampler step index is out of range");
    }
    const float sigma = schedule_.sigmas()[step_index];
    const float sigma_next = schedule_.sigmas()[step_index + 1];
    FloatTensor denoised = predicted_original_sample(
        model_output, sample, sigma, config_.prediction_type);
    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};

    if (sigma_next == 0.0F) {
        result.values = denoised.values;
    } else {
        const double ratio = static_cast<double>(sigma_next) / static_cast<double>(sigma);
        double current_coefficient = 1.0 - ratio;
        double old_coefficient = 0.0;
        if (old_denoised_.has_value()) {
            require_same_shape(*old_denoised_, denoised, "DPM++ 2M history");
            const float sigma_previous = schedule_.sigmas()[step_index - 1];
            const double time = -std::log(static_cast<double>(sigma));
            const double time_next = -std::log(static_cast<double>(sigma_next));
            const double time_previous = -std::log(static_cast<double>(sigma_previous));
            const double step_size = time_next - time;
            const double previous_step_size = time - time_previous;
            const double history_ratio = previous_step_size / step_size;
            if (!(history_ratio > 0.0) || !std::isfinite(history_ratio)) {
                throw Error("DPM++ 2M encountered an invalid sigma history ratio");
            }
            current_coefficient *= 1.0 + 1.0 / (2.0 * history_ratio);
            old_coefficient = -(1.0 - ratio) / (2.0 * history_ratio);
        }
        for (std::size_t index = 0; index < sample.values.size(); ++index) {
            const double old_value = old_denoised_.has_value()
                ? static_cast<double>(old_denoised_->values[index]) : 0.0;
            result.values[index] = static_cast<float>(
                ratio * static_cast<double>(sample.values[index]) +
                current_coefficient * static_cast<double>(denoised.values[index]) +
                old_coefficient * old_value);
        }
    }
    old_denoised_ = std::move(denoised);
    return result;
}

DDIMScheduler::DDIMScheduler(SchedulerConfig config)
    : config_(config), schedule_(config) {}

void DDIMScheduler::set_timesteps(std::size_t inference_steps,
                                  SchedulerKind scheduler) {
    schedule_.set_timesteps(inference_steps, scheduler);
}

FloatTensor DDIMScheduler::scale_model_input(const FloatTensor& sample,
                                             std::size_t step_index) const {
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("DDIM sampler step index is out of range");
    }
    return sample;
}

FloatTensor DDIMScheduler::step(const FloatTensor& model_output,
                                std::size_t step_index,
                                const FloatTensor& sample,
                                float eta,
                                std::mt19937_64* generator) const {
    require_same_shape(model_output, sample, "DDIM sampler step");
    if (step_index >= schedule_.timesteps().size()) {
        throw Error("DDIM sampler step index is out of range");
    }
    if (eta < 0.0F) throw Error("DDIM eta cannot be negative");

    const float sigma = schedule_.sigmas()[step_index];
    const float sigma_next = schedule_.sigmas()[step_index + 1];
    const float alpha_t = 1.0F / (sigma * sigma + 1.0F);
    const float alpha_previous = sigma_next == 0.0F
        ? schedule_.final_alpha_cumprod()
        : 1.0F / (sigma_next * sigma_next + 1.0F);
    const float beta_t = 1.0F - alpha_t;
    const float sqrt_alpha_t = std::sqrt(alpha_t);
    const float sqrt_beta_t = std::sqrt(beta_t);

    const float variance = std::max(
        0.0F,
        ((1.0F - alpha_previous) / (1.0F - alpha_t)) *
            (1.0F - alpha_t / alpha_previous));
    const float standard_deviation = eta * std::sqrt(variance);
    const float direction_scale = std::sqrt(
        std::max(0.0F, 1.0F - alpha_previous -
                       standard_deviation * standard_deviation));

    std::vector<float> noise;
    if (standard_deviation > 0.0F) {
        if (generator == nullptr) throw Error("DDIM eta > 0 requires a random generator");
        noise.resize(sample.values.size());
        fill_legacy_normal(noise, *generator);
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
            predicted_epsilon = sqrt_beta_t > 0.0F
                ? (current - sqrt_alpha_t * predicted_original) / sqrt_beta_t : 0.0F;
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
    // torch.manual_seed on CPU initializes MT19937 from the low 32
    // seed bits. Matching the current PyTorch Float32 randn fill preserves the
    // same initial latent for ComfyUI workflow parity.
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    fill_pytorch_cpu_normal(result.values, generator);
    return result;
}

} // namespace sdxl
