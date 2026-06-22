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
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--tokenizer-dir") {
                if (index + 1 >= argc) throw std::runtime_error("--tokenizer-dir requires a path");
                tokenizer_override = std::filesystem::path(argv[++index]);
            } else {
                positional.push_back(argument);
            }
        }
        if (positional.size() < 8) {
            std::cout
                << "Usage: sdxl_denoise <weights> <width> <height> <steps> <guidance> "
                   "<seed> <euler|ddim> <prompt> [negative] [output.sdxlf32] "
                   "[--tokenizer-dir <path>]\n";
            return 0;
        }

        const std::filesystem::path weights_path = positional[0];
        const std::size_t width = static_cast<std::size_t>(std::stoull(positional[1]));
        const std::size_t height = static_cast<std::size_t>(std::stoull(positional[2]));
        const std::size_t steps = static_cast<std::size_t>(std::stoull(positional[3]));
        const float guidance = std::stof(positional[4]);
        const std::uint64_t seed = std::stoull(positional[5]);
        const std::string scheduler_name = positional[6];
        const std::string prompt = positional[7];
        const std::string negative = positional.size() >= 9 ? positional[8] : "";
        const std::filesystem::path output_path =
            positional.size() >= 10 ? positional[9] : "latents.sdxlf32";

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
        options.guidance_scale = guidance;
        options.seed = seed;
        options.scheduler = scheduler_name == "ddim"
                                ? sdxl::SchedulerKind::DDIM
                                : sdxl::SchedulerKind::EulerDiscrete;
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
