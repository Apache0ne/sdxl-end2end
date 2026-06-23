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

[[nodiscard]] int prediction_type_code(PredictionType type) {
    switch (type) {
    case PredictionType::Epsilon: return 0;
    case PredictionType::Sample: return 1;
    case PredictionType::VPrediction: return 2;
    }
    return 0;
}

[[nodiscard]] ScalarType sampler_state_type(const SamplerConfig& config) {
    return config.state_precision == SamplerStatePrecision::Float32
        ? ScalarType::Float32 : ScalarType::Float16;
}

[[nodiscard]] float initial_noise_scale(const DenoiseOptions& options) {
    if (options.sampler == SamplerKind::DDIM) return 1.0F;
    SigmaSchedule schedule(options.scheduler_config);
    schedule.set_timesteps(options.inference_steps, options.scheduler);
    const float sigma = schedule.initial_noise_sigma();
    const float model_sigma_max = schedule.training_sigmas().back();
    const float scale = std::max({1.0F, std::abs(sigma), std::abs(model_sigma_max)});
    const bool max_denoise = sigma > model_sigma_max ||
        std::abs(sigma - model_sigma_max) <= 1.0e-5F * scale;
    if (max_denoise &&
        options.sampler_config.initial_noise_scaling ==
            InitialNoiseScaling::ComfyUIMaxDenoise) {
        // ComfyUI ModelSamplingDiscrete::noise_scaling(..., max_denoise=true).
        return std::sqrt(1.0F + sigma * sigma);
    }
    return sigma;
}

[[nodiscard]] FloatTensor host_random_normal_batch(
    const std::vector<std::size_t>& shape,
    const std::vector<std::uint64_t>& seeds,
    float scale) {
    if (shape.empty() || seeds.empty() || shape.front() != seeds.size()) {
        throw CudaError("CPU noise generation requires one seed per batch item");
    }
    std::vector<std::size_t> item_shape = shape;
    item_shape.front() = 1;
    const std::size_t per_batch = element_count(item_shape);
    FloatTensor result{shape, std::vector<float>(per_batch * seeds.size())};
    for (std::size_t batch_index = 0; batch_index < seeds.size(); ++batch_index) {
        FloatTensor item = random_normal_tensor(item_shape, seeds[batch_index]);
        std::transform(item.values.begin(), item.values.end(),
                       result.values.begin() + static_cast<std::ptrdiff_t>(batch_index * per_batch),
                       [scale](float value) { return value * scale; });
    }
    return result;
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
    const float scale = initial_noise_scale(options);
    const ScalarType state_type = sampler_state_type(options.sampler_config);
    const std::vector<std::size_t> shape{batch, 4, options.height / 8, options.width / 8};
    if (!options.batch_seeds.empty() && options.batch_seeds.size() != batch) {
        throw CudaError("batch seed count must match the prompt batch");
    }

    if (options.sampler_config.noise_device == NoiseDevice::CPU) {
        FloatTensor host;
        if (options.batch_seeds.empty()) {
            host = random_normal_tensor(shape, options.seed);
            for (float& value : host.values) value *= scale;
        } else {
            host = host_random_normal_batch(shape, options.batch_seeds, scale);
        }
        return Tensor::from_host_f32(
            *runtime_, host, state_type, TensorRole::SamplerState);
    }

    Tensor latents = Tensor::allocate(
        *runtime_, shape, state_type, TensorRole::SamplerState);
    if (!options.batch_seeds.empty()) {
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
    const std::vector<std::size_t> expected_latent_shape{batch, 4, options.height / 8, options.width / 8};
    const ScalarType expected_state_type = sampler_state_type(options.sampler_config);
    if (latents.type() != expected_state_type || latents.role() != TensorRole::SamplerState ||
        latents.shape() != expected_latent_shape) throw CudaError("initial latent tensor has the wrong shape, precision, or role");
    const std::size_t conditioning_multiplier = conditioning.classifier_free ? 2 : 1;
    if (time_ids.type() != ScalarType::Float16 || time_ids.role() != TensorRole::Model ||
        time_ids.shape() != std::vector<std::size_t>{batch * conditioning_multiplier, 6}) {
        throw CudaError("SDXL time-ID tensor has the wrong shape, precision, or role");
    }
    if (!std::isfinite(options.sampler_config.eta) || options.sampler_config.eta < 0.0F ||
        !std::isfinite(options.sampler_config.s_noise) || options.sampler_config.s_noise < 0.0F ||
        !std::isfinite(options.sampler_config.r) || options.sampler_config.r <= 0.0F ||
        options.sampler_config.r > 1.0F || !std::isfinite(options.sampler_config.s_churn) ||
        options.sampler_config.s_churn < 0.0F || !std::isfinite(options.sampler_config.s_tmin) ||
        !std::isfinite(options.sampler_config.s_tmax) || options.sampler_config.s_tmax < options.sampler_config.s_tmin) throw CudaError("sampler controls are invalid");

    const int prediction = prediction_type_code(options.scheduler_config.prediction_type);
    SigmaSchedule schedule(options.scheduler_config);
    schedule.set_timesteps(options.inference_steps, options.scheduler);
    const auto& timesteps = schedule.timesteps();
    const auto& sigmas = schedule.sigmas();
    std::vector<float> step_milliseconds;
    if (options.profile_steps) step_milliseconds.reserve(timesteps.size());

    struct Predictions { Tensor guided; Tensor unconditional; };
    auto predict = [&](const Tensor& sample, float sigma, float timestep, bool need_unconditional) {
        Tensor model_input = conditioning.classifier_free
            ? ops_.euler_scale_repeat_input(sample, sigma, 2)
            : ops_.euler_scale_input(sample, sigma);
        Tensor batch_prediction = unet_.forward(model_input, timestep, conditioning.prompt_embeds,
                                                 conditioning.pooled_text_embeds, time_ids);
        Predictions result;
        if (conditioning.classifier_free) {
            result.unconditional = ops_.batch_slice(batch_prediction, 0, batch);
            result.guided = ops_.classifier_free_guidance(batch_prediction, batch,
                                                          options.guidance_scale,
                                                          options.guidance_rescale);
        } else {
            if (need_unconditional) throw CudaError("CFG++ sampler requires unconditional conditioning");
            result.guided = batch_prediction;
            result.unconditional = batch_prediction;
        }
        return result;
    };
    auto noise_from_seeds = [&](const std::vector<std::uint64_t>& seeds, float scale) {
        if (seeds.empty()) throw CudaError("noise seed list cannot be empty");
        if (seeds.size() == 1) {
            if (options.sampler_config.noise_device == NoiseDevice::GPU) {
                return ops_.random_normal(
                    latents.shape(), seeds.front(), scale, latents.type(), latents.role());
            }
            FloatTensor host = random_normal_tensor(latents.shape(), seeds.front());
            for (float& value : host.values) value *= scale;
            return Tensor::from_host_f32(
                *runtime_, host, latents.type(), TensorRole::SamplerState);
        }
        if (seeds.size() != batch) {
            throw CudaError("noise seed count must be one or match the latent batch");
        }
        if (options.sampler_config.noise_device == NoiseDevice::GPU) {
            Tensor result = Tensor::allocate(
                *runtime_, latents.shape(), latents.type(), TensorRole::SamplerState);
            ops_.random_normal_batch_into(result, seeds, scale);
            return result;
        }
        FloatTensor host = host_random_normal_batch(latents.shape(), seeds, scale);
        return Tensor::from_host_f32(
            *runtime_, host, latents.type(), TensorRole::SamplerState);
    };
    auto noise_from_stream = [&](std::uint64_t stream, float scale) {
        if (options.batch_seeds.empty()) {
            return noise_from_seeds({options.seed ^ stream}, scale);
        }
        std::vector<std::uint64_t> seeds(options.batch_seeds.size());
        for (std::size_t index = 0; index < seeds.size(); ++index) {
            seeds[index] = options.batch_seeds[index] ^ stream;
        }
        return noise_from_seeds(seeds, scale);
    };
    auto noise = [&](std::size_t step, std::uint64_t stage, float scale) {
        const std::uint64_t stream =
            (0x9E3779B97F4A7C15ULL * (step + 1)) ^ stage;
        return noise_from_stream(stream, scale);
    };

    if (options.sampler == SamplerKind::DPMpp2M) {
        Tensor old_denoised;
        for (std::size_t step = 0; step < timesteps.size(); ++step) {
            std::optional<CudaEventTimer> timer; if (options.profile_steps) timer.emplace(*runtime_);
            Predictions outputs = predict(latents, sigmas[step], timesteps[step], false);
            Tensor denoised = ops_.predicted_original(outputs.guided, latents, sigmas[step], prediction);
            Tensor next = ops_.dpmpp_2m_step(denoised, latents,
                old_denoised.defined() ? &old_denoised : nullptr,
                sigmas[step], sigmas[step+1], step > 0 ? sigmas[step-1] : 0.0F);
            old_denoised = std::move(denoised); latents = std::move(next);
            if (timer) step_milliseconds.push_back(timer->stop());
        }
    } else if (options.sampler == SamplerKind::Euler) {
        for (std::size_t step=0; step<timesteps.size(); ++step) {
            std::optional<CudaEventTimer> timer; if(options.profile_steps) timer.emplace(*runtime_);
            const float sigma=sigmas[step];
            const float gamma=options.sampler_config.s_churn>0.0F && sigma>=options.sampler_config.s_tmin && sigma<=options.sampler_config.s_tmax
                ? std::min(options.sampler_config.s_churn/static_cast<float>(timesteps.size()),std::sqrt(2.0F)-1.0F) : 0.0F;
            const float sigma_hat=sigma*(1.0F+gamma);
            if(gamma>0.0F){
                Tensor n=noise(step,0xECULL,1.0F);
                const float ns=std::sqrt(std::max(0.0F,sigma_hat*sigma_hat-sigma*sigma))*options.sampler_config.s_noise;
                latents=ops_.combine(latents,1.0F,nullptr,0.0F,nullptr,0.0F,&n,ns);
            }
            Predictions outputs=predict(latents,sigma_hat,gamma>0.0F?schedule.timestep_for(sigma_hat):timesteps[step],false);
            latents=ops_.euler_step(outputs.guided,latents,sigma_hat,sigmas[step+1],prediction);
            if(timer) step_milliseconds.push_back(timer->stop());
        }
    } else if (options.sampler == SamplerKind::EulerAncestral) {
        for(std::size_t step=0;step<timesteps.size();++step){
            Predictions outputs=predict(latents,sigmas[step],timesteps[step],false);
            Tensor denoised=ops_.predicted_original(outputs.guided,latents,sigmas[step],prediction);
            auto a=make_ancestral_step(sigmas[step],sigmas[step+1],options.sampler_config.eta);
            if(a.down==0.0F) latents=std::move(denoised);
            else {
                const float cs=a.down/sigmas[step], cd=1.0F-cs;
                Tensor n; if(a.up>0 && options.sampler_config.s_noise>0) n=noise(step,0xEAULL,1.0F);
                latents=ops_.combine(latents,cs,&denoised,cd,nullptr,0.0F,n.defined()?&n:nullptr,a.up*options.sampler_config.s_noise);
            }
        }
    } else if (options.sampler == SamplerKind::DPMppSDE) {
        std::vector<DPMppSDEStage> stages(timesteps.size());
        std::vector<float> brownian_points;
        brownian_points.reserve(timesteps.size() * 3);
        for (std::size_t step = 0; step < timesteps.size(); ++step) {
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

        auto brownian_noise = [&](float sigma_from, float sigma_to) {
            if (!brownian_plan.has_value()) {
                throw CudaError("DPM++ SDE Brownian interval plan is unavailable");
            }
            const std::vector<BrownianIntervalWeight> weights =
                brownian_plan->weights(sigma_from, sigma_to);
            Tensor result;
            for (const BrownianIntervalWeight& weight : weights) {
                std::vector<std::uint64_t> seeds;
                if (options.batch_seeds.empty()) {
                    seeds.push_back(brownian_interval_seed(
                        options.seed, weight.interval_index));
                } else {
                    seeds.reserve(options.batch_seeds.size());
                    for (const std::uint64_t seed : options.batch_seeds) {
                        seeds.push_back(brownian_interval_seed(
                            seed, weight.interval_index));
                    }
                }
                Tensor component = noise_from_seeds(seeds, 1.0F);
                if (!result.defined()) {
                    result = ops_.combine(component, weight.coefficient);
                } else {
                    result = ops_.combine(
                        result, 1.0F, &component, weight.coefficient);
                }
            }
            return result;
        };

        for (std::size_t step = 0; step < timesteps.size(); ++step) {
            const float sigma = sigmas[step];
            const float sigma_next = sigmas[step + 1];
            Predictions first_prediction = predict(
                latents, sigma, timesteps[step], false);
            Tensor first_denoised = ops_.predicted_original(
                first_prediction.guided, latents, sigma, prediction);
            if (sigma_next == 0.0F) {
                latents = std::move(first_denoised);
                continue;
            }

            const DPMppSDEStage& stage = stages[step];
            Tensor midpoint_noise;
            if (stage.midpoint.up > 0.0F &&
                options.sampler_config.s_noise > 0.0F) {
                midpoint_noise = brownian_noise(sigma, stage.sigma_mid);
            }
            Tensor midpoint = ops_.combine(
                latents, stage.midpoint_sample_coefficient,
                &first_denoised, stage.midpoint_denoised_coefficient,
                nullptr, 0.0F,
                midpoint_noise.defined() ? &midpoint_noise : nullptr,
                stage.midpoint.up * options.sampler_config.s_noise);

            Predictions second_prediction = predict(
                midpoint, stage.sigma_mid,
                schedule.timestep_for(stage.sigma_mid), false);
            Tensor second_denoised = ops_.predicted_original(
                second_prediction.guided, midpoint, stage.sigma_mid, prediction);
            Tensor denoised_mix = ops_.combine(
                first_denoised, stage.first_denoised_mix,
                &second_denoised, stage.second_denoised_mix);

            Tensor final_noise;
            if (stage.final.up > 0.0F &&
                options.sampler_config.s_noise > 0.0F) {
                final_noise = brownian_noise(sigma, sigma_next);
            }
            latents = ops_.combine(
                latents, stage.final_sample_coefficient,
                &denoised_mix, stage.final_denoised_coefficient,
                nullptr, 0.0F,
                final_noise.defined() ? &final_noise : nullptr,
                stage.final.up * options.sampler_config.s_noise);
        }
    } else if (options.sampler == SamplerKind::DPMpp2SAncestralCFGpp) {
        for(std::size_t step=0;step<timesteps.size();++step){
            const float sigma=sigmas[step], sigma_next=sigmas[step+1];
            Predictions p1=predict(latents,sigma,timesteps[step],true);
            Tensor guided=ops_.predicted_original(p1.guided,latents,sigma,prediction);
            Tensor uncond=ops_.predicted_original(p1.unconditional,latents,sigma,prediction);
            auto a=make_ancestral_step(sigma,sigma_next,options.sampler_config.eta);
            if(a.down==0.0F){latents=std::move(guided);continue;}
            const double t=-std::log(sigma), tn=-std::log(a.down), h=tn-t, sm=std::exp(-(t+0.5*h));
            Tensor corrected=ops_.combine(latents,1.0F,&guided,1.0F,&uncond,-1.0F);
            Tensor x2=ops_.combine(corrected,static_cast<float>(sm/sigma),&guided,
                                   static_cast<float>(-std::expm1(-0.5*h)));
            Predictions p2=predict(x2,static_cast<float>(sm),schedule.timestep_for(static_cast<float>(sm)),true);
            Tensor guided2=ops_.predicted_original(p2.guided,x2,static_cast<float>(sm),prediction);
            Tensor n; if(a.up>0 && options.sampler_config.s_noise>0)n=noise(step,0x2AULL,1.0F);
            latents=ops_.combine(corrected,static_cast<float>(a.down/sigma),&guided2,
                                 static_cast<float>(-std::expm1(-h)),nullptr,0.0F,
                                 n.defined()?&n:nullptr,a.up*options.sampler_config.s_noise);
        }
    } else {
        for (std::size_t step = 0; step < timesteps.size(); ++step) {
            Tensor model_input_base = latents.type() == ScalarType::Float16
                ? latents : ops_.cast(latents, ScalarType::Float16, TensorRole::Model);
            Tensor model_input = conditioning.classifier_free
                ? ops_.repeat_batch(model_input_base, 2) : std::move(model_input_base);
            Tensor prediction_batch=unet_.forward(model_input,timesteps[step],conditioning.prompt_embeds,
                                                   conditioning.pooled_text_embeds,time_ids);
            const float sigma=sigmas[step], sigma_next=sigmas[step+1];
            const float alpha_t=1.0F/(sigma*sigma+1.0F);
            const float alpha_previous=sigma_next==0.0F
                ? schedule.final_alpha_cumprod()
                : 1.0F/(sigma_next*sigma_next+1.0F);
            Tensor n; if(options.ddim_eta>0)n=noise(step,0xDD1ULL,1.0F);
            if(!conditioning.classifier_free) latents=ops_.ddim_step(prediction_batch,latents,alpha_t,alpha_previous,options.ddim_eta,prediction,n.defined()?&n:nullptr);
            else {
                Tensor guided=ops_.classifier_free_guidance(prediction_batch,batch,options.guidance_scale,options.guidance_rescale);
                latents=ops_.ddim_step(guided,latents,alpha_t,alpha_previous,options.ddim_eta,prediction,n.defined()?&n:nullptr);
            }
        }
    }
    return DenoiseResult{std::move(latents), timesteps, std::move(step_milliseconds)};
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
    if (output_latents.type() != sampler_state_type(options.sampler_config) ||
        output_latents.role() != TensorRole::SamplerState ||
        output_latents.shape() != expected) {
        throw CudaError("graph output latent tensor has the wrong shape, precision, or role");
    }
    DenoiseResult result = run_loop(
        conditioning, initial_latents, time_ids, std::move(options));
    output_latents.copy_from(*runtime_, result.latents);
}

} // namespace sdxl::cuda
