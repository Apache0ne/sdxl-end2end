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


[[nodiscard]] FloatTensor scaled_input(const FloatTensor& sample, float sigma) {
    FloatTensor result{sample.shape, sample.values};
    const float scale = 1.0F / std::sqrt(1.0F + sigma * sigma);
    for (float& value : result.values) value *= scale;
    return result;
}

[[nodiscard]] FloatTensor predicted_original(const FloatTensor& prediction,
                                             const FloatTensor& sample,
                                             float sigma,
                                             PredictionType type) {
    if (prediction.shape != sample.shape || prediction.values.size() != sample.values.size()) {
        throw Error("prediction and sample shapes differ");
    }
    FloatTensor result{sample.shape, std::vector<float>(sample.values.size())};
    const float alpha = 1.0F / std::sqrt(1.0F + sigma * sigma);
    const float beta = sigma * alpha;
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        switch (type) {
        case PredictionType::Epsilon:
            result.values[i] = sample.values[i] - sigma * prediction.values[i];
            break;
        case PredictionType::Sample:
            result.values[i] = prediction.values[i];
            break;
        case PredictionType::VPrediction:
            result.values[i] = alpha * sample.values[i] - beta * prediction.values[i];
            break;
        }
    }
    return result;
}


void add_scaled_noise(FloatTensor& sample, const FloatTensor& noise, float scale) {
    if (sample.shape != noise.shape || sample.values.size() != noise.values.size()) {
        throw Error("noise and sample shapes differ");
    }
    for (std::size_t i = 0; i < sample.values.size(); ++i) sample.values[i] += scale * noise.values[i];
}

[[nodiscard]] FloatTensor deterministic_noise(const std::vector<std::size_t>& shape,
                                              std::uint64_t seed,
                                              std::size_t step,
                                              std::uint64_t stage) {
    return random_normal_tensor(shape, seed ^ (0x9E3779B97F4A7C15ULL * (step + 1)) ^ stage);
}

[[nodiscard]] FloatTensor brownian_noise(const BrownianIntervalPlan& plan,
                                           const std::vector<std::size_t>& shape,
                                           std::uint64_t seed,
                                           float sigma_from,
                                           float sigma_to) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) count *= dimension;
    FloatTensor result{shape, std::vector<float>(count, 0.0F)};
    for (const BrownianIntervalWeight& weight : plan.weights(sigma_from, sigma_to)) {
        const FloatTensor component = random_normal_tensor(
            shape, brownian_interval_seed(seed, weight.interval_index));
        for (std::size_t index = 0; index < result.values.size(); ++index) {
            result.values[index] += weight.coefficient * component.values[index];
        }
    }
    return result;
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
    if (options.sampler != SamplerKind::DDIM) {
        SigmaSchedule schedule(options.scheduler_config);
        schedule.set_timesteps(options.inference_steps, options.scheduler);
        initial_sigma = schedule.initial_noise_sigma();
        const float model_sigma_max = schedule.training_sigmas().back();
        const float comparison_scale = std::max(
            {1.0F, std::abs(initial_sigma), std::abs(model_sigma_max)});
        const bool max_denoise = initial_sigma > model_sigma_max ||
            std::abs(initial_sigma - model_sigma_max) <= 1.0e-5F * comparison_scale;
        if (max_denoise && options.sampler_config.initial_noise_scaling ==
                InitialNoiseScaling::ComfyUIMaxDenoise) {
            initial_sigma = std::sqrt(1.0F + initial_sigma * initial_sigma);
        }
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

    if (!std::isfinite(options.sampler_config.eta) || options.sampler_config.eta < 0.0F ||
        !std::isfinite(options.sampler_config.s_noise) || options.sampler_config.s_noise < 0.0F ||
        !std::isfinite(options.sampler_config.r) || options.sampler_config.r <= 0.0F ||
        options.sampler_config.r > 1.0F || !std::isfinite(options.sampler_config.s_churn) ||
        options.sampler_config.s_churn < 0.0F || !std::isfinite(options.sampler_config.s_tmin) ||
        !std::isfinite(options.sampler_config.s_tmax) || options.sampler_config.s_tmax < options.sampler_config.s_tmin) {
        throw Error("sampler controls are invalid");
    }

    struct Predictions { FloatTensor conditional; FloatTensor unconditional; FloatTensor guided; };
    auto predict = [&](const FloatTensor& sample, float sigma, float timestep, bool require_unconditional) {
        FloatTensor input = scaled_input(sample, sigma);
        Predictions output;
        output.conditional = unet_.forward(input, timestep, positive_prompt, positive_pooled, positive_time_ids);
        if (require_unconditional || options.guidance_scale > 1.0F) {
            output.unconditional = unet_.forward(input, timestep,
                conditioning.negative.prompt_embeds,
                conditioning.negative.pooled_prompt_embeds,
                negative_time_ids);
            output.guided = classifier_free_guidance(output.unconditional, output.conditional,
                                                     options.guidance_scale);
            apply_guidance_rescale(output.guided, output.conditional, options.guidance_rescale);
        } else {
            output.unconditional = output.conditional;
            output.guided = output.conditional;
        }
        return output;
    };

    SigmaSchedule sigma_schedule(options.scheduler_config);
    sigma_schedule.set_timesteps(options.inference_steps, options.scheduler);
    used_timesteps = sigma_schedule.timesteps();
    const auto& sigmas = sigma_schedule.sigmas();

    if (options.sampler == SamplerKind::DPMpp2M) {
        DPMpp2MSampler sampler(options.scheduler_config);
        sampler.set_timesteps(options.inference_steps, options.scheduler);
        used_timesteps = sampler.timesteps();
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            FloatTensor input = sampler.scale_model_input(latents, step);
            Predictions p = predict(latents, sampler.sigmas()[step], used_timesteps[step], false);
            (void)input;
            latents = sampler.step(p.guided, step, latents);
            if (options.progress) options.progress(step + 1, used_timesteps.size(), used_timesteps[step], latents);
        }
    } else if (options.sampler == SamplerKind::Euler) {
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float sigma = sigmas[step];
            const float gamma = options.sampler_config.s_churn > 0.0F &&
                sigma >= options.sampler_config.s_tmin && sigma <= options.sampler_config.s_tmax
                ? std::min(options.sampler_config.s_churn / static_cast<float>(used_timesteps.size()),
                           std::sqrt(2.0F) - 1.0F) : 0.0F;
            const float sigma_hat = sigma * (1.0F + gamma);
            if (gamma > 0.0F) {
                const float noise_scale = std::sqrt(std::max(0.0F, sigma_hat * sigma_hat - sigma * sigma)) *
                                          options.sampler_config.s_noise;
                add_scaled_noise(latents, deterministic_noise(latents.shape, options.seed, step, 0xECULL), noise_scale);
            }
            Predictions p = predict(latents, sigma_hat, gamma > 0.0F ? sigma_schedule.timestep_for(sigma_hat) : used_timesteps[step], false);
            FloatTensor denoised = predicted_original(p.guided, latents, sigma_hat, options.scheduler_config.prediction_type);
            for (std::size_t i=0;i<latents.values.size();++i) {
                const float d=(latents.values[i]-denoised.values[i])/sigma_hat;
                latents.values[i] += d*(sigmas[step+1]-sigma_hat);
            }
            if (options.progress) options.progress(step + 1, used_timesteps.size(), used_timesteps[step], latents);
        }
    } else if (options.sampler == SamplerKind::EulerAncestral) {
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float sigma = sigmas[step], sigma_next = sigmas[step + 1];
            Predictions p = predict(latents, sigma, used_timesteps[step], false);
            FloatTensor denoised = predicted_original(p.guided, latents, sigma, options.scheduler_config.prediction_type);
            const auto ancestral = make_ancestral_step(sigma, sigma_next, options.sampler_config.eta);
            if (ancestral.down == 0.0F) latents = std::move(denoised);
            else {
                for (std::size_t i=0;i<latents.values.size();++i) {
                    const float d=(latents.values[i]-denoised.values[i])/sigma;
                    latents.values[i] += d*(ancestral.down-sigma);
                }
                if (ancestral.up > 0.0F && options.sampler_config.s_noise > 0.0F) {
                    add_scaled_noise(latents, deterministic_noise(latents.shape, options.seed, step, 0xEAULL),
                                     ancestral.up*options.sampler_config.s_noise);
                }
            }
            if (options.progress) options.progress(step + 1, used_timesteps.size(), used_timesteps[step], latents);
        }
    } else if (options.sampler == SamplerKind::DPMppSDE) {
        std::vector<DPMppSDEStage> stages(used_timesteps.size());
        std::vector<float> brownian_points;
        brownian_points.reserve(used_timesteps.size() * 3);
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float sigma = sigmas[step];
            const float sigma_next = sigmas[step + 1];
            if (sigma_next == 0.0F) continue;
            stages[step] = make_dpmpp_sde_stage(
                sigma, sigma_next, options.sampler_config.eta,
                options.sampler_config.r);
            brownian_points.push_back(sigma);
            brownian_points.push_back(stages[step].sigma_mid);
            brownian_points.push_back(sigma_next);
        }
        std::optional<BrownianIntervalPlan> brownian_plan;
        if (brownian_points.size() >= 2) {
            brownian_plan.emplace(std::move(brownian_points));
        }

        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            const float sigma = sigmas[step];
            const float sigma_next = sigmas[step + 1];
            Predictions first_prediction = predict(
                latents, sigma, used_timesteps[step], false);
            FloatTensor first_denoised = predicted_original(
                first_prediction.guided, latents, sigma,
                options.scheduler_config.prediction_type);
            if (sigma_next == 0.0F) {
                latents = std::move(first_denoised);
            } else {
                const DPMppSDEStage& stage = stages[step];
                FloatTensor midpoint{
                    latents.shape, std::vector<float>(latents.values.size())};
                for (std::size_t index = 0; index < midpoint.values.size(); ++index) {
                    midpoint.values[index] =
                        stage.midpoint_sample_coefficient * latents.values[index] +
                        stage.midpoint_denoised_coefficient * first_denoised.values[index];
                }
                if (stage.midpoint.up > 0.0F &&
                    options.sampler_config.s_noise > 0.0F) {
                    if (!brownian_plan.has_value()) {
                        throw Error("DPM++ SDE Brownian interval plan is unavailable");
                    }
                    add_scaled_noise(
                        midpoint,
                        brownian_noise(*brownian_plan, midpoint.shape, options.seed,
                                       sigma, stage.sigma_mid),
                        stage.midpoint.up * options.sampler_config.s_noise);
                }

                Predictions second_prediction = predict(
                    midpoint, stage.sigma_mid,
                    sigma_schedule.timestep_for(stage.sigma_mid), false);
                FloatTensor second_denoised = predicted_original(
                    second_prediction.guided, midpoint, stage.sigma_mid,
                    options.scheduler_config.prediction_type);

                for (std::size_t index = 0; index < latents.values.size(); ++index) {
                    const float denoised_mix =
                        stage.first_denoised_mix * first_denoised.values[index] +
                        stage.second_denoised_mix * second_denoised.values[index];
                    latents.values[index] =
                        stage.final_sample_coefficient * latents.values[index] +
                        stage.final_denoised_coefficient * denoised_mix;
                }
                if (stage.final.up > 0.0F &&
                    options.sampler_config.s_noise > 0.0F) {
                    if (!brownian_plan.has_value()) {
                        throw Error("DPM++ SDE Brownian interval plan is unavailable");
                    }
                    add_scaled_noise(
                        latents,
                        brownian_noise(*brownian_plan, latents.shape, options.seed,
                                       sigma, sigma_next),
                        stage.final.up * options.sampler_config.s_noise);
                }
            }
            if (options.progress) {
                options.progress(step + 1, used_timesteps.size(),
                                 used_timesteps[step], latents);
            }
        }
    } else if (options.sampler == SamplerKind::DPMpp2SAncestralCFGpp) {
        for (std::size_t step=0; step<used_timesteps.size(); ++step) {
            const float sigma=sigmas[step], sigma_next=sigmas[step+1];
            Predictions p1=predict(latents,sigma,used_timesteps[step],true);
            FloatTensor guided=predicted_original(p1.guided,latents,sigma,options.scheduler_config.prediction_type);
            FloatTensor uncond=predicted_original(p1.unconditional,latents,sigma,options.scheduler_config.prediction_type);
            auto a=make_ancestral_step(sigma,sigma_next,options.sampler_config.eta);
            if(a.down==0.0F) latents=std::move(guided);
            else {
                const double t=-std::log(sigma), tn=-std::log(a.down), h=tn-t, r=0.5, sm=std::exp(-(t+r*h));
                FloatTensor x2{latents.shape,std::vector<float>(latents.values.size())};
                const double ratio=sm/sigma, c=-std::expm1(-h*r);
                for(size_t i=0;i<x2.values.size();++i) x2.values[i]=static_cast<float>(ratio*(latents.values[i]+guided.values[i]-uncond.values[i])+c*guided.values[i]);
                Predictions p2=predict(x2,static_cast<float>(sm),static_cast<float>(0.5*(used_timesteps[step]+(step+1<used_timesteps.size()?used_timesteps[step+1]:0.0F))),true);
                FloatTensor guided2=predicted_original(p2.guided,x2,static_cast<float>(sm),options.scheduler_config.prediction_type);
                const double ratio2=a.down/sigma, c2=-std::expm1(-h);
                for(size_t i=0;i<latents.values.size();++i) latents.values[i]=static_cast<float>(ratio2*(latents.values[i]+guided.values[i]-uncond.values[i])+c2*guided2.values[i]);
                if(a.up>0 && options.sampler_config.s_noise>0) add_scaled_noise(latents,deterministic_noise(latents.shape,options.seed,step,0x2AULL),a.up*options.sampler_config.s_noise);
            }
            if(options.progress) options.progress(step+1,used_timesteps.size(),used_timesteps[step],latents);
        }
    } else {
        DDIMScheduler sampler(options.scheduler_config);
        sampler.set_timesteps(options.inference_steps, options.scheduler);
        used_timesteps = sampler.timesteps();
        for (std::size_t step = 0; step < used_timesteps.size(); ++step) {
            Predictions p = predict(latents, sampler.sigmas()[step], used_timesteps[step], false);
            latents = sampler.step(p.guided, step, latents, options.ddim_eta, &ddim_generator);
            if (options.progress) options.progress(step + 1, used_timesteps.size(), used_timesteps[step], latents);
        }
    }

    return SDXLDenoiseResult{std::move(latents), std::move(used_timesteps)};
}

} // namespace sdxl
