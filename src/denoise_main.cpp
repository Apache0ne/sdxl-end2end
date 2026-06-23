#include "sdxl/denoiser.hpp"
#include "sdxl/sdxl.hpp"
#include "sdxl/text_encoder.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void write_float_tensor(const sdxl::FloatTensor& tensor,
                        const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output file: " + path.string());
    const std::uint32_t magic = 0x4C585453U;
    const std::uint32_t rank = static_cast<std::uint32_t>(tensor.shape.size());
    output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (const std::size_t dimension : tensor.shape) {
        const std::uint64_t value = static_cast<std::uint64_t>(dimension);
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    output.write(reinterpret_cast<const char*>(tensor.values.data()),
                 static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed while writing output file");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> positional;
        std::optional<std::filesystem::path> tokenizer_override;
        std::string sampler_name = "dpmpp_2m";
        std::string scheduler_name = "normal";
        bool sampler_explicit = false;
        bool hyper_sdxl = false;
        float ddim_eta = 0.0F;
        sdxl::SamplerConfig sampler_config;
        sdxl::SchedulerConfig scheduler_config;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--tokenizer-dir") {
                if (index + 1 >= argc) throw std::runtime_error("--tokenizer-dir requires a path");
                tokenizer_override = std::filesystem::path(argv[++index]);
            } else if (argument == "--hyper-sdxl") {
                hyper_sdxl = true;
            } else if (argument == "--sampler") {
                if (index + 1 >= argc) throw std::runtime_error("--sampler requires a value");
                sampler_name = argv[++index];
                sampler_explicit = true;
            } else if (argument == "--scheduler") {
                if (index + 1 >= argc) throw std::runtime_error("--scheduler requires a value");
                scheduler_name = argv[++index];
            } else if (argument == "--ddim-set-alpha-to-one") {
                scheduler_config.set_alpha_to_one = true;
            } else if (argument == "--ddim-eta") {
                if (index + 1 >= argc) throw std::runtime_error("--ddim-eta requires a value");
                ddim_eta = std::stof(argv[++index]);
            } else if (argument == "--eta" || argument == "--sampler-eta") {
                sampler_config.eta = std::stof(argv[++index]);
            } else if (argument == "--s-noise" || argument == "--sampler-s-noise") {
                sampler_config.s_noise = std::stof(argv[++index]);
            } else if (argument == "--r" || argument == "--sampler-r") {
                sampler_config.r = std::stof(argv[++index]);
            } else if (argument == "--s-churn") {
                sampler_config.s_churn = std::stof(argv[++index]);
            } else if (argument == "--s-tmin") {
                sampler_config.s_tmin = std::stof(argv[++index]);
            } else if (argument == "--s-tmax") {
                sampler_config.s_tmax = std::stof(argv[++index]);
            } else if (argument == "--noise-device") {
                sampler_config.noise_device = sdxl::parse_noise_device(argv[++index]);
            } else if (argument == "--coeff" || argument == "--gits-coeff") {
                scheduler_config.gits_coeff = std::stof(argv[++index]);
            } else if (argument == "--denoise" || argument == "--scheduler-denoise") {
                scheduler_config.denoise = std::stof(argv[++index]);
            } else if (argument == "--alpha" || argument == "--beta-alpha") {
                scheduler_config.beta_schedule_alpha = std::stof(argv[++index]);
            } else if (argument == "--beta" || argument == "--beta-beta") {
                scheduler_config.beta_schedule_beta = std::stof(argv[++index]);
            } else if (argument == "--karras-rho") {
                scheduler_config.karras_rho = std::stof(argv[++index]);
            } else if (argument == "--linear-quadratic-threshold") {
                scheduler_config.linear_quadratic_threshold = std::stof(argv[++index]);
            } else if (argument == "--training-timesteps") {
                scheduler_config.training_timesteps = std::stoull(argv[++index]);
            } else if (argument == "--beta-start") {
                scheduler_config.beta_start = std::stof(argv[++index]);
            } else if (argument == "--beta-end") {
                scheduler_config.beta_end = std::stof(argv[++index]);
            } else {
                positional.push_back(argument);
            }
        }
        if (positional.size() < 7) {
            std::cout
                << "Usage: sdxl_denoise <weights> <width> <height> <steps> <guidance> "
                   "<seed> <prompt> [negative] [output.sdxlf32] "
                   "[--sampler dpmpp_2m|dpmpp_sde|euler|euler_ancestral|dpmpp_2s_ancestral_cfg_pp|ddim] "
                   "[--scheduler normal|karras|exponential|sgm_uniform|simple|ddim_uniform|ddim_trailing|beta|linear_quadratic|kl_optimal|gits] "
                   "[--hyper-sdxl] [--tokenizer-dir <path>]\n";
            return 0;
        }

        if (hyper_sdxl) {
            sampler_name = "ddim";
            scheduler_name = "ddim_trailing";
            ddim_eta = 0.0F;
            scheduler_config.set_alpha_to_one = false;
            sampler_explicit = true;
        }

        const std::filesystem::path weights_path = positional[0];
        const std::size_t width = static_cast<std::size_t>(std::stoull(positional[1]));
        const std::size_t height = static_cast<std::size_t>(std::stoull(positional[2]));
        const std::size_t steps = static_cast<std::size_t>(std::stoull(positional[3]));
        if (hyper_sdxl && steps != 2U && steps != 4U && steps != 8U) {
            throw std::runtime_error(
                "--hyper-sdxl is the fixed 2/4/8-step recipe; steps must be 2, 4, or 8");
        }
        const float guidance = std::stof(positional[4]);
        const float effective_guidance = hyper_sdxl ? 0.0F : guidance;
        const std::uint64_t seed = std::stoull(positional[5]);
        std::size_t prompt_index = 6;
        if (!sampler_explicit && positional.size() > prompt_index &&
            sdxl::is_sampler_kind_name(positional[prompt_index])) {
            sampler_name = positional[prompt_index++];
        }
        if (positional.size() <= prompt_index) throw std::runtime_error("missing prompt");
        const std::string prompt = positional[prompt_index];
        const std::string negative = positional.size() > prompt_index + 1
            ? positional[prompt_index + 1] : "";
        const std::filesystem::path output_path = positional.size() > prompt_index + 2
            ? positional[prompt_index + 2] : "latents.sdxlf32";

        sdxl::SDXLModel model;
        sdxl::SDXLWeightLoader loader;
        const sdxl::LoadResult loaded = loader.load(model, weights_path);
        std::cout << "Bound " << loaded.parameters_bound << " parameters from "
                  << loaded.files_mapped << " file(s).\n";

        sdxl::SDXLTextConditioner conditioner = tokenizer_override.has_value()
            ? sdxl::SDXLTextConditioner::from_model_directory(model, *tokenizer_override, {})
            : sdxl::SDXLTextConditioner::builtin_sdxl(model, {});
        sdxl::SDXLClassifierFreeConditioning conditioning =
            conditioner.encode_classifier_free({prompt}, {negative});

        sdxl::SDXLDenoiser denoiser(model);
        sdxl::SDXLDenoiseOptions options;
        options.width = width;
        options.height = height;
        options.inference_steps = steps;
        options.guidance_scale = effective_guidance;
        options.seed = seed;
        options.sampler = sdxl::parse_sampler_kind(sampler_name);
        options.scheduler = sdxl::parse_scheduler_kind(scheduler_name);
        options.sampler_config = sampler_config;
        options.ddim_eta = ddim_eta;
        options.scheduler_config = scheduler_config;
        options.progress = [](std::size_t step,
                              std::size_t total,
                              float timestep,
                              const sdxl::FloatTensor&) {
            std::cout << "Denoising " << step << '/' << total
                      << " timestep=" << timestep << '\n';
        };

        sdxl::SDXLDenoiseResult result = denoiser.denoise(conditioning, options);
        write_float_tensor(result.latents, output_path);
        std::cout << "Saved final latents to " << output_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
