#include "sdxl/sdxl.hpp"
#include "sdxl/text_encoder.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

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
        if (positional.size() < 2) {
            std::cout << "Usage: sdxl_text_encode <model-directory|checkpoint.safetensors> "
                         "<prompt...> [--tokenizer-dir <path>]\n"
                         "The standard SDXL tokenizers are embedded by default.\n";
            return 0;
        }

        const std::filesystem::path weights_path = positional.front();
        std::string prompt = positional[1];
        for (std::size_t index = 2; index < positional.size(); ++index) {
            prompt.push_back(' ');
            prompt += positional[index];
        }

        sdxl::SDXLModel model;
        sdxl::SDXLWeightLoader loader;
        const sdxl::LoadResult loaded = loader.load(model, weights_path);
        std::cout << "Loaded " << loaded.parameters_bound << " SDXL parameter tensors.\n";

        sdxl::TextEncoderExecutionOptions options;
        options.progress = [](std::string_view component, std::size_t layer, std::size_t count) {
            if (layer < count) {
                std::cout << component << " layer " << (layer + 1) << '/' << count << '\n';
            } else {
                std::cout << component << " complete\n";
            }
        };
        sdxl::SDXLTextConditioner conditioner = tokenizer_override.has_value()
            ? sdxl::SDXLTextConditioner::from_model_directory(model, *tokenizer_override, options)
            : sdxl::SDXLTextConditioner::builtin_sdxl(model, options);
        sdxl::SDXLPromptConditioning conditioning = conditioner.encode(prompt);

        std::cout << "CLIP-L tokens: " << conditioning.clip_l_tokens.sequence_length << '\n';
        std::cout << "OpenCLIP tokens: " << conditioning.openclip_tokens.sequence_length << '\n';
        std::cout << "Prompt embeddings shape: [";
        for (std::size_t index = 0; index < conditioning.prompt_embeds.shape.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << conditioning.prompt_embeds.shape[index];
        }
        std::cout << "]\nPooled embeddings shape: [";
        for (std::size_t index = 0; index < conditioning.pooled_prompt_embeds.shape.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << conditioning.pooled_prompt_embeds.shape[index];
        }
        std::cout << "]\nFirst prompt embedding values:";
        const std::size_t sample_count = std::min<std::size_t>(8, conditioning.prompt_embeds.values.size());
        for (std::size_t index = 0; index < sample_count; ++index) {
            std::cout << ' ' << conditioning.prompt_embeds.values[index];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
