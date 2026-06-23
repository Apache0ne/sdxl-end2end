#include "sdxl/scheduler.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float tolerance = 1.0e-4F) {
    return std::abs(a - b) <= tolerance;
}

} // namespace

int main() {
    try {
        const sdxl::SamplerConfig defaults;
        if (defaults.noise_device != sdxl::NoiseDevice::CPU ||
            defaults.state_precision != sdxl::SamplerStatePrecision::Float32 ||
            defaults.initial_noise_scaling !=
                sdxl::InitialNoiseScaling::ComfyUIMaxDenoise) {
            return 1;
        }
        if (sdxl::parse_sampler_state_precision("comfyui") !=
                sdxl::SamplerStatePrecision::Float32 ||
            sdxl::parse_initial_noise_scaling("max-denoise") !=
                sdxl::InitialNoiseScaling::ComfyUIMaxDenoise) {
            return 2;
        }

        const sdxl::FloatTensor pytorch_noise =
            sdxl::random_normal_tensor({1, 4, 2, 2}, 1234);
        const std::vector<float> pytorch_reference{
            -0.111718565F, -0.496590108F, 0.163073704F, -0.881687760F,
             0.053900193F,  0.668373704F, -0.059657607F, -0.467497945F,
            -0.215252936F,  0.883961618F, -0.758416891F, -0.368867904F,
            -0.342393607F, -1.401971817F,  0.320647985F, -1.021857500F};
        if (pytorch_noise.values.size() != pytorch_reference.size()) return 11;
        for (std::size_t index = 0; index < pytorch_reference.size(); ++index) {
            if (!near(pytorch_noise.values[index], pytorch_reference[index], 2.0e-6F)) {
                return 12;
            }
        }

        sdxl::SigmaSchedule normal;
        normal.set_timesteps(4, sdxl::SchedulerKind::Normal);
        const std::vector<float> expected_timesteps{999.0F, 666.0F, 333.0F, 0.0F};
        const std::vector<float> expected_sigmas{
            14.614641F, 2.918307F, 0.932358F, 0.029167F, 0.0F};
        if (normal.timesteps().size() != expected_timesteps.size() ||
            normal.sigmas().size() != expected_sigmas.size()) {
            return 3;
        }
        for (std::size_t i = 0; i < expected_timesteps.size(); ++i) {
            if (!near(normal.timesteps()[i], expected_timesteps[i], 2.0e-4F) ||
                !near(normal.sigmas()[i], expected_sigmas[i], 3.0e-3F)) {
                return 4;
            }
        }
        const auto& training = normal.training_sigmas();
        const std::size_t lower_index = 400;
        const float log_midpoint_sigma = static_cast<float>(std::exp(
            0.5 * (std::log(static_cast<double>(training[lower_index])) +
                   std::log(static_cast<double>(training[lower_index + 1])))));
        if (!near(normal.timestep_for(log_midpoint_sigma),
                  static_cast<float>(lower_index), 1.0e-6F)) {
            return 10;
        }

        sdxl::SchedulerConfig partial_config;
        partial_config.denoise = 0.5F;
        sdxl::SigmaSchedule partial(partial_config);
        partial.set_timesteps(4, sdxl::SchedulerKind::Normal);
        sdxl::SchedulerConfig full_config;
        sdxl::SigmaSchedule full(full_config);
        full.set_timesteps(8, sdxl::SchedulerKind::Normal);
        if (partial.sigmas().size() != 5 || partial.timesteps().size() != 4) return 5;
        for (std::size_t i = 0; i < partial.sigmas().size(); ++i) {
            if (!near(partial.sigmas()[i], full.sigmas()[i + 4], 1.0e-6F)) return 6;
        }
        for (std::size_t i = 0; i < partial.timesteps().size(); ++i) {
            if (!near(partial.timesteps()[i], full.timesteps()[i + 4], 1.0e-6F)) return 7;
        }

        const sdxl::DPMppSDEStage stage =
            sdxl::make_dpmpp_sde_stage(
                normal.sigmas()[0], normal.sigmas()[1], 1.0F, 0.5F);
        if (!near(stage.sigma_mid, 6.530697F, 3.0e-5F) ||
            !near(stage.midpoint.up, 5.842388F, 3.0e-5F) ||
            !near(stage.final.up, 2.859533F, 3.0e-5F)) {
            return 8;
        }
        const sdxl::BrownianIntervalPlan tree({
            normal.sigmas()[1], stage.sigma_mid, normal.sigmas()[0]});
        const auto midpoint = tree.weights(normal.sigmas()[0], stage.sigma_mid);
        const auto complete = tree.weights(normal.sigmas()[0], normal.sigmas()[1]);
        if (midpoint.empty() || complete.size() != 2) return 9;

        std::cout << "ComfyUI scheduler and DPM++ SDE parity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 100;
    }
}
