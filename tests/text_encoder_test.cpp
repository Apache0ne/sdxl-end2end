#include "sdxl/text_encoder.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using Storage = std::map<std::string, std::vector<float>>;

std::vector<std::int64_t> strides(const std::vector<std::uint64_t>& shape) {
    std::vector<std::int64_t> result(shape.size());
    std::int64_t stride = static_cast<std::int64_t>(sizeof(float));
    for (std::size_t index = shape.size(); index-- > 0;) {
        result[index] = stride;
        stride *= static_cast<std::int64_t>(shape[index]);
    }
    return result;
}

void bind_component(sdxl::SDXLModel& model, const std::string& component, Storage& storage) {
    for (auto& [name, slot] : model.graph().parameter_index()) {
        if (name.rfind(component + ".", 0) != 0) continue;
        std::size_t count = 1;
        for (const auto dimension : slot->expected_shape) count *= static_cast<std::size_t>(dimension);
        auto& values = storage[name];
        values.resize(count);
        const bool norm_weight = name.find("layer_norm") != std::string::npos && name.ends_with(".weight");
        const bool final_norm_weight = name.find("final_layer_norm.weight") != std::string::npos;
        const bool bias = name.ends_with(".bias");
        if (norm_weight || final_norm_weight) {
            std::fill(values.begin(), values.end(), 1.0F);
        } else if (bias) {
            std::fill(values.begin(), values.end(), 0.0F);
        } else if (name.ends_with("text_projection.weight") && slot->expected_shape.size() == 2) {
            std::fill(values.begin(), values.end(), 0.0F);
            const std::size_t rows = static_cast<std::size_t>(slot->expected_shape[0]);
            const std::size_t columns = static_cast<std::size_t>(slot->expected_shape[1]);
            for (std::size_t index = 0; index < std::min(rows, columns); ++index) {
                values[index * columns + index] = 1.0F;
            }
        } else {
            for (std::size_t index = 0; index < count; ++index) {
                const int centered = static_cast<int>(index % 17) - 8;
                values[index] = static_cast<float>(centered) * 0.002F;
            }
        }
        sdxl::TensorView view;
        view.data = reinterpret_cast<const std::byte*>(values.data());
        view.dtype = sdxl::DType::F32;
        view.shape = slot->expected_shape;
        view.strides_bytes = strides(view.shape);
        view.storage_bytes = values.size() * sizeof(float);
        view.source_key = name;
        slot->tensor = std::move(view);
    }
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
}

bool all_finite(const std::vector<float>& values) {
    for (const float value : values) if (!std::isfinite(value)) return false;
    return true;
}

} // namespace

int main() {
    try {
        sdxl::SDXLConfig config = sdxl::SDXLConfig::base_1_0();
        config.clip_l = {16, 5, 4, 8, 2, 2, 4, false, "quick_gelu"};
        config.openclip_big_g = {16, 5, 4, 8, 2, 2, 4, true, "gelu"};
        sdxl::SDXLModel model(config);
        Storage storage;
        bind_component(model, "text_encoder", storage);
        bind_component(model, "text_encoder_2", storage);

        const auto root = std::filesystem::temp_directory_path() / "sdxl_raw_text_test";
        std::filesystem::remove_all(root);
        for (const char* name : {"tokenizer", "tokenizer_2"}) {
            const auto directory = root / name;
            std::filesystem::create_directories(directory);
            write_file(directory / "vocab.json",
                       R"({"<|startoftext|>":14,"<|endoftext|>":15,"a</w>":2,"!</w>":3,"c":4,"a":5,"t</w>":6,"cat</w>":7})");
            write_file(directory / "merges.txt", "#version: 0.2\nc a\nca t</w>\n");
            write_file(directory / "tokenizer_config.json", R"({"model_max_length":5})");
        }

        sdxl::TextEncoderExecutionOptions options;
        options.thread_count = 2;
        options.linear_output_block = 2;
        const auto conditioner = sdxl::SDXLTextConditioner::from_model_directory(model, root, options);
        const auto output = conditioner.encode("A cat!");
        if (output.prompt_embeds.shape != std::vector<std::size_t>{1, 5, 8}) return 1;
        if (output.pooled_prompt_embeds.shape != std::vector<std::size_t>{1, 4}) return 2;
        if (!all_finite(output.prompt_embeds.values) || !all_finite(output.pooled_prompt_embeds.values)) return 3;
        if (output.clip_l_tokens.input_ids != std::vector<std::int32_t>{14, 2, 7, 3, 15}) return 4;
        std::filesystem::remove_all(root);
        std::cout << "text encoder test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 5;
    }
}
