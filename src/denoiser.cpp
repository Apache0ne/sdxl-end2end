#include "sdxl/denoiser.hpp"

#include "sdxl/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>

namespace sdxl {
namespace {

void validate_conditioning_tensor(const FloatTensor& prompt,
                                  const FloatTensor& pooled,
                                  std::size_t expected_batch,
                                  const SDXLConfig& config,
                                  const char* label) {
    if (prompt.shape.size() != 3 || prompt.shape[0] != expected_batch ||
        prompt.shape[2] != config.unet_cross_attention_dim ||
        prompt.values.size() != prompt.shape[0] * prompt.shape[1] * prompt.shape[2]) {
        throw Error(std::string(label) + " prompt embeddings have an invalid shape");
    }
    const std::size_t pooled_width = static_cast<std::size_t>(
        config.unet_addition_input_dim - 6 * config.unet_addition_time_embed_dim);
    if (pooled.shape != std::vector<std::size_t>{expected_batch, pooled_width} ||
        pooled.values.size() != expected_batch * pooled_width) {
        throw Error(std::string(label) + " pooled prompt embeddings have an invalid shape");
    }
}

[[nodiscard]] SDXLMicroConditioning resolve_micro(SDXLMicroConditioning micro,
                                                  std::size_t height,
                                                  std::size_t width) {
    if (height > std::numeric_limits<std::uint32_t>::max() ||
        width > std::numeric_limits<std::uint32_t>::max()) {
        throw Error("requested image dimensions exceed SDXL time-ID range");
    }
    if (micro.original_height == 0) micro.original_height = static_cast<std::uint32_t>(height);
    if (micro.original_width == 0) micro.original_width = static_cast<std::uint32_t>(width);
    if (micro.target_height == 0) micro.target_height = static_cast<std::uint32_t>(height);
    if (micro.target_width == 0) micro.target_width = static_cast<std::uint32_t>(width);
    return micro;
}

[[nodiscard]] FloatTensor classifier_free_guidance(const FloatTensor& unconditional,
                                                    const FloatTensor& conditional,
                                                    float guidance_scale) {
    if (unconditional.shape != conditional.shape ||
        unconditional.values.size() != conditional.values.size()) {
        throw Error("classifier-free guidance predictions have different shapes");
    }
    FloatTensor result{unconditional.shape,
                       std::vector<float>(unconditional.values.size())};
    for (std::size_t index = 0; index < result.values.size(); ++index) {
        result.values[index] = unconditional.values[index] +
                               guidance_scale *
                                   (conditional.values[index] - unconditional.values[index]);
    }
    return result;
}

[[nodiscard]] double batch_standard_deviation(const FloatTensor& tensor,
                                              std::size_t batch_index) {
    if (tensor.shape.empty() || batch_index >= tensor.shape[0]) {
        throw Error("standard-deviation batch index is invalid");
    }
    const std::size_t batch_size = tensor.shape[0];
    if (batch_size == 0 || tensor.values.size() % batch_size != 0) {
        throw Error("tensor batch storage is inconsistent");
    }
    const std::size_t count = tensor.values.size() / batch_size;
    const float* values = tensor.values.data() + batch_index * count;
    double mean = 0.0;
    for (std::size_t index = 0; index < count; ++index) mean += values[index];
    mean /= static_cast<double>(count);
    double squared = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double centered = static_cast<double>(values[index]) - mean;
        squared += centered * centered;
    }
    const double denominator = static_cast<double>(count > 1 ? count - 1 : 1);
    return std::sqrt(squared / denominator);
}

void apply_guidance_rescale(FloatTensor& guided,
                            const FloatTensor& conditional,
                            float amount) {
    if (amount <= 0.0F) return;
    if (amount > 1.0F) throw Error("guidance rescale must be between zero and one");
    if (guided.shape != conditional.shape || guided.values.size() != conditional.values.size()) {
        throw Error("guidance-rescale tensors have different shapes");
    }
    const std::size_t batch = guided.shape[0];
    const std::size_t per_batch = guided.values.size() / batch;
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        const double conditional_std = batch_standard_deviation(conditional, batch_index);
        const double guided_std = batch_standard_deviation(guided, batch_index);
        const float ratio = guided_std > 0.0
                                ? static_cast<float>(conditional_std / guided_std)
                                : 1.0F;
        float* values = guided.values.data() + batch_index * per_batch;
        for (std::size_t index = 0; index < per_batch; ++index) {
            const float rescaled = values[index] * ratio;
            values[index] = amount * rescaled + (1.0F - amount) * values[index];
        }
    }
}

void validate_initial_latents(const FloatTensor& latents,
                              std::size_t batch,
                              std::size_t height,
                              std::size_t width) {
    const std::vector<std::size_t> expected{batch, 4, height / 8, width / 8};
    if (latents.shape != expected) {
        throw Error("initial latent shape must be [B,4,height/8,width/8]");
    }
    const std::size_t count = batch * 4 * (height / 8) * (width / 8);
    if (latents.values.size() != count) throw Error("initial latent storage is inconsistent");
}

} // namespace

SDXLDenoiser::SDXLDenoiser(const SDXLModel& model,
                           UNetExecutionOptions unet_options)
    : model_(&model), unet_(model, std::move(unet_options)) {}

SDXLDenoiseResult SDXLDenoiser::denoise(
    const SDXLClassifierFreeConditioning& conditioning,
    SDXLDenoiseOptions options) const {
    if (conditioning.positive.prompt_embeds.shape.empty()) {
        throw Error("positive SDXL conditioning is empty");
    }
    const std::size_t batch = conditioning.positive.prompt_embeds.shape[0];
    if (options.width == 0 || options.height == 0 ||
        options.width % 8 != 0 || options.height % 8 != 0) {
        throw Error("SDXL image width and height must be positive multiples of eight");
    }
    FloatTensor initial = random_normal_tensor(
        {batch, 4, options.height / 8, options.width / 8}, options.seed);

    float initial_sigma = 1.0F;
    if (options.scheduler == SchedulerKind::EulerDiscrete) {
        EulerDiscreteScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        initial_sigma = scheduler.initial_noise_sigma();
    }
    for (float& value : initial.values) value *= initial_sigma;
    return denoise_from_latents(conditioning, std::move(initial), std::move(options));
}

SDXLDenoiseResult SDXLDenoiser::denoise_from_latents(
    const SDXLClassifierFreeConditioning& conditioning,
    FloatTensor initial_latents,
    SDXLDenoiseOptions options) const {
    if (model_ == nullptr) throw Error("SDXL denoiser has no model");
    if (options.inference_steps == 0) throw Error("inference step count must be positive");
    if (options.width == 0 || options.height == 0 ||
        options.width % 8 != 0 || options.height % 8 != 0) {
        throw Error("SDXL image width and height must be positive multiples of eight");
    }
    if (!std::isfinite(options.guidance_scale) || options.guidance_scale < 0.0F) {
        throw Error("guidance scale must be finite and nonnegative");
    }
    if (!std::isfinite(options.guidance_rescale) ||
        options.guidance_rescale < 0.0F || options.guidance_rescale > 1.0F) {
        throw Error("guidance rescale must be in [0,1]");
    }

    const FloatTensor& positive_prompt = conditioning.positive.prompt_embeds;
    const FloatTensor& positive_pooled = conditioning.positive.pooled_prompt_embeds;
    if (positive_prompt.shape.empty()) throw Error("positive SDXL conditioning is empty");
    const std::size_t batch = positive_prompt.shape[0];
    validate_conditioning_tensor(positive_prompt,
                                 positive_pooled,
                                 batch,
                                 model_->config(),
                                 "positive");
    validate_conditioning_tensor(conditioning.negative.prompt_embeds,
                                 conditioning.negative.pooled_prompt_embeds,
                                 batch,
                                 model_->config(),
                                 "negative");
    validate_initial_latents(initial_latents, batch, options.height, options.width);

    const SDXLMicroConditioning positive_micro =
        resolve_micro(options.positive_micro, options.height, options.width);
    const SDXLMicroConditioning negative_micro = resolve_micro(
        options.negative_micro.value_or(positive_micro), options.height, options.width);
    const FloatTensor positive_time_ids = positive_micro.time_ids(batch);
    const FloatTensor negative_time_ids = negative_micro.time_ids(batch);

    FloatTensor latents = std::move(initial_latents);
    std::vector<float> used_timesteps;
    std::mt19937_64 ddim_generator(options.seed ^ 0x9E3779B97F4A7C15ULL);

    if (options.scheduler == SchedulerKind::EulerDiscrete) {
        EulerDiscreteScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        used_timesteps = scheduler.timesteps();
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float timestep = used_timesteps[step];
            FloatTensor model_input = scheduler.scale_model_input(latents, step);
            FloatTensor conditional = unet_.forward(model_input,
                                                    timestep,
                                                    positive_prompt,
                                                    positive_pooled,
                                                    positive_time_ids);
            FloatTensor prediction;
            if (options.guidance_scale > 1.0F) {
                FloatTensor unconditional = unet_.forward(
                    model_input,
                    timestep,
                    conditioning.negative.prompt_embeds,
                    conditioning.negative.pooled_prompt_embeds,
                    negative_time_ids);
                prediction = classifier_free_guidance(
                    unconditional, conditional, options.guidance_scale);
                apply_guidance_rescale(prediction, conditional, options.guidance_rescale);
            } else {
                prediction = std::move(conditional);
            }
            latents = scheduler.step(prediction, step, latents);
            if (options.progress) options.progress(step + 1, used_timesteps.size(), timestep, latents);
        }
    } else {
        DDIMScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        used_timesteps = scheduler.timesteps();
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float timestep = used_timesteps[step];
            FloatTensor model_input = scheduler.scale_model_input(latents, step);
            FloatTensor conditional = unet_.forward(model_input,
                                                    timestep,
                                                    positive_prompt,
                                                    positive_pooled,
                                                    positive_time_ids);
            FloatTensor prediction;
            if (options.guidance_scale > 1.0F) {
                FloatTensor unconditional = unet_.forward(
                    model_input,
                    timestep,
                    conditioning.negative.prompt_embeds,
                    conditioning.negative.pooled_prompt_embeds,
                    negative_time_ids);
                prediction = classifier_free_guidance(
                    unconditional, conditional, options.guidance_scale);
                apply_guidance_rescale(prediction, conditional, options.guidance_rescale);
            } else {
                prediction = std::move(conditional);
            }
            latents = scheduler.step(prediction,
                                     step,
                                     latents,
                                     options.ddim_eta,
                                     &ddim_generator);
            if (options.progress) options.progress(step + 1, used_timesteps.size(), timestep, latents);
        }
    }

    return SDXLDenoiseResult{std::move(latents), std::move(used_timesteps)};
}

} // namespace sdxl
