#include "sdxl/denoiser.hpp"
#include "sdxl/model.hpp"
#include "sdxl/scheduler.hpp"
#include "sdxl/sdxl.hpp"
#include "sdxl/unet.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace {

class OwnedWeights final {
public:
    void bind_unet(sdxl::SDXLModel& model) {
        for (auto& [name, slot] : model.graph().parameter_index()) {
            if (name.rfind("unet.", 0) != 0) continue;
            std::size_t count = 1;
            for (const std::uint64_t dimension : slot->expected_shape) {
                count *= static_cast<std::size_t>(dimension);
            }
            auto storage = std::make_unique<std::vector<float>>(count, 0.0F);
            if (name.ends_with(".weight") &&
                (name.find(".norm") != std::string::npos ||
                 name.find("conv_norm_out.weight") != std::string::npos)) {
                std::fill(storage->begin(), storage->end(), 1.0F);
            }
            if (name == "unet.conv_out.bias") {
                std::fill(storage->begin(), storage->end(), 0.25F);
            }

            sdxl::TensorView view;
            view.data = reinterpret_cast<const std::byte*>(storage->data());
            view.dtype = sdxl::DType::F32;
            view.shape = slot->expected_shape;
            view.strides_bytes.resize(view.shape.size());
            std::int64_t stride = static_cast<std::int64_t>(sizeof(float));
            for (std::size_t reverse = view.shape.size(); reverse > 0; --reverse) {
                const std::size_t dimension = reverse - 1;
                view.strides_bytes[dimension] = stride;
                stride *= static_cast<std::int64_t>(view.shape[dimension]);
            }
            view.storage_bytes = static_cast<std::uint64_t>(count * sizeof(float));
            view.source_key = name;
            slot->tensor = view;
            storage_.push_back(std::move(storage));
        }
    }

private:
    std::list<std::unique_ptr<std::vector<float>>> storage_;
};

sdxl::SDXLConfig miniature_config() {
    sdxl::SDXLConfig config;
    config.clip_l = {16, 4, 8, 16, 1, 1, 8, false, "quick_gelu"};
    config.openclip_big_g = {16, 4, 8, 16, 1, 1, 8, true, "gelu"};
    config.unet_channels = {32, 32, 32};
    config.unet_heads = {1, 1, 1};
    config.unet_transformer_depth = {1, 1, 1};
    config.unet_cross_attention_dim = 8;
    config.unet_time_embed_dim = 64;
    config.unet_addition_time_embed_dim = 4;
    config.unet_addition_input_dim = 32; // 8 pooled + 6 * 4 time features.
    config.unet_layers_per_block = 2;
    config.unet_norm_groups = 32;
    config.vae_channels = {32, 32, 32, 32};
    config.vae_latent_channels = 4;
    config.vae_layers_per_block = 2;
    return config;
}

bool near(float first, float second, float tolerance = 1.0e-5F) {
    return std::abs(first - second) <= tolerance;
}

} // namespace

int main() {
    try {
        sdxl::SDXLModel base_model;
        if (base_model.graph().parameter_index().size() != 2641) return 1;
        const sdxl::ParameterSlot* base_unet_projection =
            base_model.graph().find_parameter(
                "unet.mid_block.attentions.0.transformer_blocks.9.attn2.to_k.weight");
        if (base_unet_projection == nullptr ||
            base_unet_projection->expected_shape != std::vector<std::uint64_t>{1280, 2048}) {
            return 2;
        }

        if (sdxl::parse_sampler_kind("dpmpp_2m") != sdxl::SamplerKind::DPMpp2M ||
            sdxl::parse_sampler_kind("DPM++-2M") != sdxl::SamplerKind::DPMpp2M ||
            sdxl::parse_sampler_kind("dpmpp_sde") != sdxl::SamplerKind::DPMppSDE ||
            sdxl::parse_sampler_kind("euler") != sdxl::SamplerKind::Euler ||
            sdxl::parse_sampler_kind("SamplerEulerAncestral") != sdxl::SamplerKind::EulerAncestral ||
            sdxl::parse_sampler_kind("dpmpp_2s_ancestral_cfg_pp") != sdxl::SamplerKind::DPMpp2SAncestralCFGpp ||
            sdxl::parse_sampler_kind("ddim") != sdxl::SamplerKind::DDIM) {
            return 3;
        }
        if (sdxl::parse_scheduler_kind("normal") != sdxl::SchedulerKind::Normal ||
            sdxl::parse_scheduler_kind("sgm-uniform") != sdxl::SchedulerKind::SGMUniform ||
            sdxl::parse_scheduler_kind("ddim-trailing") != sdxl::SchedulerKind::DDIMTrailing ||
            sdxl::parse_scheduler_kind("hyper_sdxl") != sdxl::SchedulerKind::DDIMTrailing ||
            sdxl::parse_scheduler_kind("GITSScheduler") != sdxl::SchedulerKind::GITS ||
            std::string(sdxl::sampler_kind_name(sdxl::SamplerKind::DPMpp2M)) != "dpmpp_2m" ||
            std::string(sdxl::scheduler_kind_name(sdxl::SchedulerKind::KLOptimal)) != "kl_optimal") {
            return 4;
        }

        const std::vector<sdxl::SchedulerKind> scheduler_kinds{
            sdxl::SchedulerKind::Normal,
            sdxl::SchedulerKind::Karras,
            sdxl::SchedulerKind::Exponential,
            sdxl::SchedulerKind::SGMUniform,
            sdxl::SchedulerKind::Simple,
            sdxl::SchedulerKind::DDIMUniform,
            sdxl::SchedulerKind::DDIMTrailing,
            sdxl::SchedulerKind::Beta,
            sdxl::SchedulerKind::LinearQuadratic,
            sdxl::SchedulerKind::KLOptimal,
            sdxl::SchedulerKind::GITS,
        };
        const std::vector<std::vector<float>> four_step_reference{
            {14.614641F, 2.918307F, 0.932358F, 0.029167F},
            {14.614641F, 3.168603F, 0.446918F, 0.029167F},
            {14.614641F, 1.840024F, 0.231664F, 0.029167F},
            {14.614641F, 4.086081F, 1.615580F, 0.695150F},
            {14.614641F, 4.081729F, 1.612886F, 0.693205F},
            {4.116696F, 1.623692F, 0.698398F, 0.041314F},
            {14.614641F, 4.081729F, 1.612886F, 0.693205F},
            {14.614641F, 5.686591F, 1.618279F, 0.515391F},
            {14.614641F, 14.431958F, 14.249275F, 10.595615F},
            {14.614641F, 1.597066F, 0.572914F, 0.029167F},
            {14.614641F, 2.363261F, 0.921923F, 0.366170F},
        };
        for (std::size_t kind_index = 0; kind_index < scheduler_kinds.size(); ++kind_index) {
            const sdxl::SchedulerKind kind = scheduler_kinds[kind_index];
            sdxl::SigmaSchedule schedule;
            schedule.set_timesteps(20, kind);
            if (schedule.timesteps().empty() ||
                schedule.sigmas().size() != schedule.timesteps().size() + 1 ||
                schedule.sigmas().back() != 0.0F) {
                return 5;
            }
            for (std::size_t index = 0; index < schedule.timesteps().size(); ++index) {
                if (!std::isfinite(schedule.timesteps()[index]) ||
                    !std::isfinite(schedule.sigmas()[index]) ||
                    !(schedule.sigmas()[index] > 0.0F)) {
                    return 6;
                }
                if (index > 0 && !(schedule.sigmas()[index] < schedule.sigmas()[index - 1])) {
                    return 7;
                }
            }

            schedule.set_timesteps(4, kind);
            if (schedule.timesteps().size() != four_step_reference[kind_index].size()) return 8;
            for (std::size_t index = 0; index < four_step_reference[kind_index].size(); ++index) {
                const float expected = four_step_reference[kind_index][index];
                const float tolerance = std::max(2.0e-4F, std::abs(expected) * 1.0e-3F);
                if (!near(schedule.sigmas()[index], expected, tolerance)) return 9;
            }
        }

        sdxl::SigmaSchedule hyper_schedule;
        hyper_schedule.set_timesteps(4, sdxl::SchedulerKind::DDIMTrailing);
        const std::vector<float> hyper_timesteps{999.0F, 749.0F, 499.0F, 249.0F};
        if (hyper_schedule.timesteps() != hyper_timesteps) return 37;
        const float sigma_zero = hyper_schedule.training_sigmas().front();
        const float expected_final_alpha = 1.0F / (1.0F + sigma_zero * sigma_zero);
        if (!near(hyper_schedule.final_alpha_cumprod(), expected_final_alpha, 1.0e-7F) ||
            near(hyper_schedule.final_alpha_cumprod(), 1.0F, 1.0e-7F)) return 41;
        sdxl::SchedulerConfig alpha_one_config;
        alpha_one_config.set_alpha_to_one = true;
        sdxl::SigmaSchedule alpha_one_schedule(alpha_one_config);
        if (alpha_one_schedule.final_alpha_cumprod() != 1.0F) return 42;

        const auto& training_sigmas = hyper_schedule.training_sigmas();
        const std::size_t inverse_lower = 400;
        const float fractional_sigma = static_cast<float>(std::exp(
            0.5 * (std::log(static_cast<double>(training_sigmas[inverse_lower])) +
                   std::log(static_cast<double>(training_sigmas[inverse_lower + 1])))));
        if (!near(hyper_schedule.timestep_for(fractional_sigma),
                  static_cast<float>(inverse_lower), 1.0e-6F)) return 38;

        const sdxl::DPMppSDEStage sde_stage =
            sdxl::make_dpmpp_sde_stage(14.614641F, 2.918307F, 1.0F, 0.5F);
        if (!(sde_stage.sigma_mid < 14.614641F &&
              sde_stage.sigma_mid > 2.918307F) ||
            !near(sde_stage.first_denoised_mix, 0.0F, 1.0e-6F) ||
            !near(sde_stage.second_denoised_mix, 1.0F, 1.0e-6F) ||
            !near(sde_stage.midpoint_sample_coefficient +
                      sde_stage.midpoint_denoised_coefficient,
                  1.0F, 1.0e-6F) ||
            !near(sde_stage.final_sample_coefficient +
                      sde_stage.final_denoised_coefficient,
                  1.0F, 1.0e-6F) ||
            !near(sde_stage.sigma_mid, 6.530697F, 2.0e-5F) ||
            !near(sde_stage.midpoint.down, 2.918307F, 2.0e-5F) ||
            !near(sde_stage.midpoint.up, 5.842388F, 2.0e-5F) ||
            !near(sde_stage.final.down, 0.582739F, 2.0e-5F) ||
            !near(sde_stage.final.up, 2.859533F, 2.0e-5F) ||
            !near(sde_stage.midpoint_sample_coefficient, 0.199684F, 2.0e-5F) ||
            !near(sde_stage.final_sample_coefficient, 0.039874F, 2.0e-5F)) return 39;
        const sdxl::DPMppSDEStage deterministic_sde_stage =
            sdxl::make_dpmpp_sde_stage(14.614641F, 2.918307F, 0.0F, 0.5F);
        if (deterministic_sde_stage.midpoint.up != 0.0F ||
            deterministic_sde_stage.final.up != 0.0F ||
            !near(deterministic_sde_stage.final.down, 2.918307F, 1.0e-6F)) return 40;

        sdxl::SchedulerConfig gits_config;
        gits_config.gits_coeff = 1.20F;
        gits_config.denoise = 0.42F;
        sdxl::SigmaSchedule gits_partial(gits_config);
        gits_partial.set_timesteps(4, sdxl::SchedulerKind::GITS);
        if (gits_partial.timesteps().size() != 2 || gits_partial.sigmas().size() != 3 ||
            !near(gits_partial.sigmas()[0], 0.921923F, 2.0e-5F) ||
            !near(gits_partial.sigmas()[1], 0.366170F, 2.0e-5F) ||
            gits_partial.sigmas().back() != 0.0F) return 28;
        gits_config.denoise = 1.0F;
        sdxl::SigmaSchedule gits_interpolated(gits_config);
        gits_interpolated.set_timesteps(25, sdxl::SchedulerKind::GITS);
        if (gits_interpolated.timesteps().size() != 25 ||
            gits_interpolated.sigmas().size() != 26 ||
            gits_interpolated.sigmas().back() != 0.0F) return 29;

        sdxl::EulerDiscreteScheduler euler;
        euler.set_timesteps(4, sdxl::SchedulerKind::Normal);
        if (euler.timesteps().size() != 4 || euler.sigmas().size() != 5) return 10;
        if (!near(euler.initial_noise_sigma(), 14.6146F, 2.0e-3F)) return 11;
        for (std::size_t index = 1; index < euler.sigmas().size(); ++index) {
            if (euler.sigmas()[index] > euler.sigmas()[index - 1]) return 12;
        }
        sdxl::FloatTensor scheduler_sample{{1, 1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        sdxl::FloatTensor zero_prediction{{1, 1, 2, 2}, {0.0F, 0.0F, 0.0F, 0.0F}};
        const sdxl::FloatTensor unchanged = euler.step(zero_prediction, 0, scheduler_sample);
        for (std::size_t index = 0; index < unchanged.values.size(); ++index) {
            if (!near(unchanged.values[index], scheduler_sample.values[index])) return 13;
        }

        sdxl::DPMpp2MSampler dpmpp;
        dpmpp.set_timesteps(3, sdxl::SchedulerKind::Normal);
        const auto& dpm_sigmas = dpmpp.sigmas();
        sdxl::FloatTensor dpm_sample{{1, 1, 1, 2}, {1.0F, 2.0F}};
        const sdxl::FloatTensor first_prediction{{1, 1, 1, 2}, {0.5F, -0.5F}};
        const sdxl::FloatTensor first_original{{1, 1, 1, 2}, {
            dpm_sample.values[0] - dpm_sigmas[0] * first_prediction.values[0],
            dpm_sample.values[1] - dpm_sigmas[0] * first_prediction.values[1]}};
        const double first_ratio = static_cast<double>(dpm_sigmas[1]) /
                                   static_cast<double>(dpm_sigmas[0]);
        const sdxl::FloatTensor first_step = dpmpp.step(first_prediction, 0, dpm_sample);
        for (std::size_t index = 0; index < first_step.values.size(); ++index) {
            const float expected = static_cast<float>(
                first_ratio * static_cast<double>(dpm_sample.values[index]) +
                (1.0 - first_ratio) * static_cast<double>(first_original.values[index]));
            if (!near(first_step.values[index], expected, 2.0e-5F)) return 14;
        }

        const sdxl::FloatTensor second_prediction{{1, 1, 1, 2}, {-0.25F, 0.75F}};
        const sdxl::FloatTensor second_original{{1, 1, 1, 2}, {
            first_step.values[0] - dpm_sigmas[1] * second_prediction.values[0],
            first_step.values[1] - dpm_sigmas[1] * second_prediction.values[1]}};
        const double second_ratio = static_cast<double>(dpm_sigmas[2]) /
                                    static_cast<double>(dpm_sigmas[1]);
        const double current_time = -std::log(static_cast<double>(dpm_sigmas[1]));
        const double next_time = -std::log(static_cast<double>(dpm_sigmas[2]));
        const double previous_time = -std::log(static_cast<double>(dpm_sigmas[0]));
        const double history_ratio = (current_time - previous_time) /
                                     (next_time - current_time);
        const sdxl::FloatTensor second_step = dpmpp.step(second_prediction, 1, first_step);
        for (std::size_t index = 0; index < second_step.values.size(); ++index) {
            const double denoised_derivative =
                (1.0 + 1.0 / (2.0 * history_ratio)) *
                    static_cast<double>(second_original.values[index]) -
                (1.0 / (2.0 * history_ratio)) *
                    static_cast<double>(first_original.values[index]);
            const float expected = static_cast<float>(
                second_ratio * static_cast<double>(first_step.values[index]) +
                (1.0 - second_ratio) * denoised_derivative);
            if (!near(second_step.values[index], expected, 3.0e-5F)) return 15;
        }
        const sdxl::FloatTensor final_prediction{{1, 1, 1, 2}, {0.1F, 0.2F}};
        const sdxl::FloatTensor final_step = dpmpp.step(final_prediction, 2, second_step);
        for (std::size_t index = 0; index < final_step.values.size(); ++index) {
            const float expected = second_step.values[index] -
                                   dpm_sigmas[2] * final_prediction.values[index];
            if (!near(final_step.values[index], expected, 3.0e-5F)) return 16;
        }

        const sdxl::FloatTensor random_a = sdxl::random_normal_tensor({1, 4, 2, 2}, 1234);
        const sdxl::FloatTensor random_b = sdxl::random_normal_tensor({1, 4, 2, 2}, 1234);
        if (random_a.values != random_b.values) return 17;

        const sdxl::BrownianIntervalPlan brownian_plan({1.0F, 2.0F, 4.0F});
        if (brownian_plan.interval_count() != 2) return 30;
        const auto high_half = brownian_plan.weights(4.0F, 2.0F);
        const auto low_half = brownian_plan.weights(2.0F, 1.0F);
        const auto whole = brownian_plan.weights(4.0F, 1.0F);
        if (high_half.size() != 1 || high_half[0].interval_index != 1 ||
            !near(high_half[0].coefficient, -1.0F, 1.0e-6F)) return 31;
        if (low_half.size() != 1 || low_half[0].interval_index != 0 ||
            !near(low_half[0].coefficient, -1.0F, 1.0e-6F)) return 32;
        if (whole.size() != 2 || whole[0].interval_index != 0 ||
            whole[1].interval_index != 1 ||
            !near(whole[0].coefficient, -std::sqrt(1.0F / 3.0F), 1.0e-6F) ||
            !near(whole[1].coefficient, -std::sqrt(2.0F / 3.0F), 1.0e-6F)) return 33;
        const float elementary_low = 0.25F;
        const float elementary_high = -0.75F;
        const float whole_increment = whole[0].coefficient * elementary_low +
                                      whole[1].coefficient * elementary_high;
        const float reconstructed =
            (std::sqrt(1.0F) * low_half[0].coefficient * elementary_low +
             std::sqrt(2.0F) * high_half[0].coefficient * elementary_high) /
            std::sqrt(3.0F);
        if (!near(whole_increment, reconstructed, 1.0e-6F)) return 34;
        if (sdxl::brownian_interval_seed(1234, 0) !=
                sdxl::brownian_interval_seed(1234, 0) ||
            sdxl::brownian_interval_seed(1234, 0) ==
                sdxl::brownian_interval_seed(1234, 1)) return 35;
        if (sdxl::parse_noise_device("cpu") != sdxl::NoiseDevice::CPU ||
            sdxl::parse_noise_device("cuda") != sdxl::NoiseDevice::GPU) return 36;

        sdxl::DDIMScheduler ddim;
        ddim.set_timesteps(4, sdxl::SchedulerKind::Normal);
        const sdxl::FloatTensor ddim_output = ddim.step(
            zero_prediction, 0, scheduler_sample, 0.0F, nullptr);
        if (ddim_output.shape != scheduler_sample.shape) return 18;
        for (const float value : ddim_output.values) {
            if (!std::isfinite(value)) return 19;
        }

        sdxl::SDXLModel model(miniature_config());
        OwnedWeights weights;
        weights.bind_unet(model);

        sdxl::SDXLUNet unet(model);
        sdxl::FloatTensor latent{{1, 4, 4, 4}, std::vector<float>(64, 0.0F)};
        sdxl::FloatTensor encoder{{1, 2, 8}, std::vector<float>(16, 0.0F)};
        sdxl::FloatTensor pooled{{1, 8}, std::vector<float>(8, 0.0F)};
        sdxl::SDXLMicroConditioning micro;
        micro.original_height = 32;
        micro.original_width = 32;
        micro.target_height = 32;
        micro.target_width = 32;
        const sdxl::FloatTensor output = unet.forward(
            latent, 999.0F, encoder, pooled, micro.time_ids(1));
        if (output.shape != latent.shape) return 20;
        for (const float value : output.values) {
            if (!near(value, 0.25F, 1.0e-4F)) return 21;
        }

        sdxl::SDXLClassifierFreeConditioning conditioning;
        conditioning.positive.prompt_embeds = encoder;
        conditioning.positive.pooled_prompt_embeds = pooled;
        conditioning.negative.prompt_embeds = encoder;
        conditioning.negative.pooled_prompt_embeds = pooled;
        sdxl::SDXLDenoiseOptions options;
        if (options.sampler != sdxl::SamplerKind::DPMpp2M ||
            options.scheduler != sdxl::SchedulerKind::Normal) return 22;
        options.width = 32;
        options.height = 32;
        options.inference_steps = 1;
        options.guidance_scale = 2.0F;
        options.seed = 9;
        sdxl::SDXLDenoiser denoiser(model);
        const sdxl::SDXLDenoiseResult result = denoiser.denoise(conditioning, options);
        if (result.latents.shape != latent.shape || result.timesteps.size() != 1) return 24;
        for (const float value : result.latents.values) {
            if (!std::isfinite(value)) return 25;
        }

        // Exercise every public sampler/scheduler pairing through the complete
        // miniature denoising loop, including DPM++ 2M history at three steps.
        const std::vector<sdxl::SamplerKind> sampler_kinds{
            sdxl::SamplerKind::DPMpp2M,
            sdxl::SamplerKind::DPMppSDE,
            sdxl::SamplerKind::Euler,
            sdxl::SamplerKind::EulerAncestral,
            sdxl::SamplerKind::DPMpp2SAncestralCFGpp,
            sdxl::SamplerKind::DDIM,
        };
        for (const sdxl::SamplerKind sampler_kind : sampler_kinds) {
            for (const sdxl::SchedulerKind scheduler_kind : scheduler_kinds) {
                sdxl::SDXLDenoiseOptions combination = options;
                combination.inference_steps = 3;
                combination.guidance_scale = 1.0F;
                combination.sampler = sampler_kind;
                combination.scheduler = scheduler_kind;
                const sdxl::SDXLDenoiseResult combination_result =
                    denoiser.denoise(conditioning, combination);
                if (combination_result.timesteps.empty() ||
                    combination_result.latents.shape != latent.shape) {
                    return 26;
                }
                for (const float value : combination_result.latents.values) {
                    if (!std::isfinite(value)) return 27;
                }
            }
        }

        std::cout << "UNet, sampler, and scheduler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 100;
    }
}
