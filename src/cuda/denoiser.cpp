#include "sdxl/cuda/denoiser.hpp"
#include "sdxl/cuda/profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace sdxl::cuda {
namespace {

[[nodiscard]] SDXLMicroConditioning resolve_micro(SDXLMicroConditioning value,
                                                  std::size_t height,
                                                  std::size_t width) {
    if (height > std::numeric_limits<std::uint32_t>::max() ||
        width > std::numeric_limits<std::uint32_t>::max()) {
        throw CudaError("requested image dimensions exceed SDXL time-ID range");
    }
    if (value.original_height == 0) value.original_height = static_cast<std::uint32_t>(height);
    if (value.original_width == 0) value.original_width = static_cast<std::uint32_t>(width);
    if (value.target_height == 0) value.target_height = static_cast<std::uint32_t>(height);
    if (value.target_width == 0) value.target_width = static_cast<std::uint32_t>(width);
    return value;
}

[[nodiscard]] std::array<float, 6> micro_values(const SDXLMicroConditioning& value) {
    return {
        static_cast<float>(value.original_height),
        static_cast<float>(value.original_width),
        static_cast<float>(value.crop_top),
        static_cast<float>(value.crop_left),
        static_cast<float>(value.target_height),
        static_cast<float>(value.target_width)
    };
}

[[nodiscard]] std::vector<float> make_alphas_cumprod(const SchedulerConfig& config) {
    if (config.training_timesteps < 2) throw CudaError("scheduler requires at least two training steps");
    std::vector<float> cumulative(config.training_timesteps);
    const double start = std::sqrt(static_cast<double>(config.beta_start));
    const double end = std::sqrt(static_cast<double>(config.beta_end));
    const double denominator = static_cast<double>(config.training_timesteps - 1);
    double product = 1.0;
    for (std::size_t index = 0; index < config.training_timesteps; ++index) {
        const double fraction = static_cast<double>(index) / denominator;
        const double beta_root = start + (end - start) * fraction;
        const double beta = beta_root * beta_root;
        product *= 1.0 - beta;
        cumulative[index] = static_cast<float>(product);
    }
    return cumulative;
}

[[nodiscard]] int prediction_type_code(PredictionType type) {
    switch (type) {
    case PredictionType::Epsilon: return 0;
    case PredictionType::Sample: return 1;
    case PredictionType::VPrediction: return 2;
    }
    return 0;
}

} // namespace

Denoiser::Denoiser(const Runtime& runtime,
                   const SDXLModel& model,
                   const WeightStore& weights,
                   UNetOptions options)
    : runtime_(&runtime),
      model_(&model),
      weights_(&weights),
      ops_(runtime),
      unet_(runtime, model, weights, options) {}

void Denoiser::validate_inputs(const ClassifierFreeConditioning& conditioning,
                               const DenoiseOptions& options) const {
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA denoiser is not initialized");
    }
    const std::size_t conditioning_multiplier = conditioning.classifier_free ? 2 : 1;
    if (conditioning.batch_size == 0 ||
        conditioning.prompt_embeds.rank() != 3 ||
        conditioning.prompt_embeds.size(0) !=
            conditioning.batch_size * conditioning_multiplier ||
        conditioning.prompt_embeds.size(2) != model_->config().unet_cross_attention_dim ||
        conditioning.pooled_text_embeds.rank() != 2 ||
        conditioning.pooled_text_embeds.size(0) !=
            conditioning.batch_size * conditioning_multiplier) {
        throw CudaError("CUDA text conditioning has invalid shapes");
    }
    if (options.width == 0 || options.height == 0 ||
        options.width % 8 != 0 || options.height % 8 != 0) {
        throw CudaError("SDXL width and height must be positive multiples of eight");
    }
    if (options.inference_steps == 0) throw CudaError("inference step count must be positive");
    if (!std::isfinite(options.guidance_scale) || options.guidance_scale < 0.0F ||
        !std::isfinite(options.guidance_rescale) || options.guidance_rescale < 0.0F ||
        options.guidance_rescale > 1.0F) {
        throw CudaError("guidance values are invalid");
    }
}

Tensor Denoiser::create_time_ids(std::size_t batch, const DenoiseOptions& options,
                                 bool classifier_free) const {
    if (batch == 0) throw CudaError("time-ID batch cannot be zero");
    SDXLMicroConditioning positive = resolve_micro(
        options.positive_micro, options.height, options.width);
    SDXLMicroConditioning negative = resolve_micro(
        options.negative_micro.value_or(positive), options.height, options.width);
    const auto positive_values = micro_values(positive);
    FloatTensor host;
    if (classifier_free) {
        const auto negative_values = micro_values(negative);
        host.shape = {batch * 2, 6};
        host.values.resize(batch * 12);
        for (std::size_t item = 0; item < batch; ++item) {
            std::copy(negative_values.begin(), negative_values.end(),
                      host.values.begin() + static_cast<std::ptrdiff_t>(item * 6));
            std::copy(positive_values.begin(), positive_values.end(),
                      host.values.begin() + static_cast<std::ptrdiff_t>((batch + item) * 6));
        }
    } else {
        host.shape = {batch, 6};
        host.values.resize(batch * 6);
        for (std::size_t item = 0; item < batch; ++item) {
            std::copy(positive_values.begin(), positive_values.end(),
                      host.values.begin() + static_cast<std::ptrdiff_t>(item * 6));
        }
    }
    return Tensor::from_host_f32(*runtime_, host, ScalarType::Float16, TensorRole::Model);
}

Tensor Denoiser::create_initial_latents(const DenoiseOptions& options,
                                        std::size_t batch) const {
    if (batch == 0 || options.width == 0 || options.height == 0 ||
        options.width % 8 != 0 || options.height % 8 != 0) {
        throw CudaError("invalid latent creation dimensions");
    }
    float scale = 1.0F;
    if (options.scheduler == SchedulerKind::EulerDiscrete) {
        EulerDiscreteScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        scale = scheduler.initial_noise_sigma();
    }
    Tensor latents = Tensor::allocate(
        *runtime_, {batch, 4, options.height / 8, options.width / 8},
        ScalarType::Float16, TensorRole::Model);
    if (!options.batch_seeds.empty()) {
        if (options.batch_seeds.size() != batch) {
            throw CudaError("batch seed count must match the prompt batch");
        }
        ops_.random_normal_batch_into(latents, options.batch_seeds, scale);
    } else {
        ops_.random_normal_into(latents, options.seed, scale);
    }
    return latents;
}

DenoiseResult Denoiser::run_loop(const ClassifierFreeConditioning& conditioning,
                                 Tensor latents,
                                 const Tensor& time_ids,
                                 DenoiseOptions options) const {
    const std::size_t batch = conditioning.batch_size;
    const std::vector<std::size_t> expected_latent_shape{
        batch, 4, options.height / 8, options.width / 8};
    if (latents.type() != ScalarType::Float16 ||
        latents.role() != TensorRole::Model ||
        latents.shape() != expected_latent_shape) {
        throw CudaError("initial latent tensor has the wrong shape, precision, or role");
    }
    const std::size_t conditioning_multiplier = conditioning.classifier_free ? 2 : 1;
    if (time_ids.type() != ScalarType::Float16 ||
        time_ids.role() != TensorRole::Model ||
        time_ids.shape() !=
            std::vector<std::size_t>{batch * conditioning_multiplier, 6}) {
        throw CudaError("SDXL time-ID tensor has the wrong shape, precision, or role");
    }

    const int prediction = prediction_type_code(options.scheduler_config.prediction_type);
    if (options.scheduler == SchedulerKind::EulerDiscrete) {
        EulerDiscreteScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        const auto& timesteps = scheduler.timesteps();
        const auto& sigmas = scheduler.sigmas();
        std::vector<float> step_milliseconds;
        if (options.profile_steps) step_milliseconds.reserve(timesteps.size());
        for (std::size_t step = 0; step < timesteps.size(); ++step) {
            auto step_profile = profile_scope("denoise/euler_step/" + std::to_string(step));
            std::optional<CudaEventTimer> timer;
            if (options.profile_steps) timer.emplace(*runtime_);
            if (conditioning.classifier_free) {
                Tensor model_input = ops_.euler_scale_repeat_input(latents, sigmas[step], 2);
                Tensor prediction_batch = unet_.forward(
                    model_input, timesteps[step], conditioning.prompt_embeds,
                    conditioning.pooled_text_embeds, time_ids);
                if (options.guidance_rescale == 0.0F) {
                    latents = ops_.euler_cfg_step(
                        prediction_batch, latents, batch, options.guidance_scale,
                        sigmas[step], sigmas[step + 1], prediction);
                } else {
                    Tensor guided = ops_.classifier_free_guidance(
                        prediction_batch, batch, options.guidance_scale,
                        options.guidance_rescale);
                    latents = ops_.euler_step(
                        guided, latents, sigmas[step], sigmas[step + 1], prediction);
                }
            } else {
                Tensor model_input = ops_.euler_scale_input(latents, sigmas[step]);
                Tensor prediction_single = unet_.forward(
                    model_input, timesteps[step], conditioning.prompt_embeds,
                    conditioning.pooled_text_embeds, time_ids);
                latents = ops_.euler_step(
                    prediction_single, latents, sigmas[step], sigmas[step + 1],
                    prediction);
            }
            if (timer.has_value()) step_milliseconds.push_back(timer->stop());
        }
        return DenoiseResult{std::move(latents), timesteps, std::move(step_milliseconds)};
    }

    if (options.inference_steps > options.scheduler_config.training_timesteps) {
        throw CudaError("DDIM inference steps cannot exceed training timesteps");
    }
    const std::vector<float> alphas = make_alphas_cumprod(options.scheduler_config);
    const std::size_t ratio = options.scheduler_config.training_timesteps / options.inference_steps;
    std::vector<float> timesteps(options.inference_steps);
    for (std::size_t index = 0; index < options.inference_steps; ++index) {
        timesteps[index] = static_cast<float>((options.inference_steps - 1 - index) * ratio);
    }
    std::vector<float> step_milliseconds;
    if (options.profile_steps) step_milliseconds.reserve(timesteps.size());
    for (std::size_t step = 0; step < timesteps.size(); ++step) {
        auto step_profile = profile_scope("denoise/ddim_step/" + std::to_string(step));
        std::optional<CudaEventTimer> timer;
        if (options.profile_steps) timer.emplace(*runtime_);
        Tensor model_input = conditioning.classifier_free
            ? ops_.repeat_batch(latents, 2) : latents;
        Tensor prediction_batch = unet_.forward(
            model_input, timesteps[step], conditioning.prompt_embeds,
            conditioning.pooled_text_embeds, time_ids);
        const std::size_t timestep = static_cast<std::size_t>(std::llround(timesteps[step]));
        const std::size_t previous = step + 1 < timesteps.size()
            ? static_cast<std::size_t>(std::llround(timesteps[step + 1]))
            : 0;
        const float alpha_t = alphas.at(timestep);
        const float alpha_previous = step + 1 < timesteps.size() ? alphas.at(previous) : 1.0F;
        Tensor noise;
        if (options.ddim_eta > 0.0F) {
            noise = ops_.random_normal(latents.shape(), options.seed + step + 1, 1.0F);
        }
        if (!conditioning.classifier_free) {
            latents = ops_.ddim_step(
                prediction_batch, latents, alpha_t, alpha_previous,
                options.ddim_eta, prediction, noise.defined() ? &noise : nullptr);
        } else if (options.guidance_rescale == 0.0F) {
            latents = ops_.ddim_cfg_step(
                prediction_batch, latents, batch, options.guidance_scale,
                alpha_t, alpha_previous, options.ddim_eta, prediction,
                noise.defined() ? &noise : nullptr);
        } else {
            Tensor guided = ops_.classifier_free_guidance(
                prediction_batch, batch, options.guidance_scale,
                options.guidance_rescale);
            latents = ops_.ddim_step(
                guided, latents, alpha_t, alpha_previous, options.ddim_eta,
                prediction, noise.defined() ? &noise : nullptr);
        }
        if (timer.has_value()) step_milliseconds.push_back(timer->stop());
    }
    return DenoiseResult{std::move(latents), std::move(timesteps), std::move(step_milliseconds)};
}

DenoiseResult Denoiser::denoise(const ClassifierFreeConditioning& conditioning,
                                DenoiseOptions options) const {
    validate_inputs(conditioning, options);
    Tensor time_ids = create_time_ids(
        conditioning.batch_size, options, conditioning.classifier_free);
    Tensor latents = create_initial_latents(options, conditioning.batch_size);
    return run_loop(conditioning, std::move(latents), time_ids, std::move(options));
}

void Denoiser::denoise_into(const ClassifierFreeConditioning& conditioning,
                            const Tensor& initial_latents,
                            const Tensor& time_ids,
                            Tensor& output_latents,
                            DenoiseOptions options) const {
    validate_inputs(conditioning, options);
    if (options.profile_steps) {
        throw CudaError("per-step event profiling cannot be captured into a reusable CUDA Graph");
    }
    const std::vector<std::size_t> expected{
        conditioning.batch_size, 4, options.height / 8, options.width / 8};
    if (output_latents.type() != ScalarType::Float16 ||
        output_latents.role() != TensorRole::Model ||
        output_latents.shape() != expected) {
        throw CudaError("graph output latent tensor has the wrong shape, precision, or role");
    }
    DenoiseResult result = run_loop(
        conditioning, initial_latents, time_ids, std::move(options));
    output_latents.copy_from(*runtime_, result.latents);
}

} // namespace sdxl::cuda
