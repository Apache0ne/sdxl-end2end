#include "sdxl/sdxl.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

#ifndef SDXL_INT8_KEY_DUMP_PATH
#error "SDXL_INT8_KEY_DUMP_PATH must name the reference key inventory"
#endif

namespace {

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

} // namespace

int main() {
    try {
        const std::filesystem::path path = SDXL_INT8_KEY_DUMP_PATH;
        std::ifstream input(path);
        require(static_cast<bool>(input), "cannot open reference INT8 key inventory");

        std::size_t declared_tensor_count = 0;
        std::unordered_set<std::string> keys;
        std::string line;
        while (std::getline(input, line)) {
            if (starts_with(line, "tensor_count:")) {
                declared_tensor_count = static_cast<std::size_t>(
                    std::stoull(line.substr(std::string("tensor_count:").size())));
            } else if (!line.empty() && line.find(':') == std::string::npos) {
                keys.insert(line);
            }
        }
        require(declared_tensor_count == 4660, "unexpected declared tensor count");
        require(keys.size() == declared_tensor_count, "key inventory count mismatch");

        sdxl::SDXLModel model;
        std::size_t clip_l_parameters = 0;
        std::size_t openclip_parameters = 0;
        for (const auto& [logical_name, slot] : model.graph().parameter_index()) {
            (void)slot;
            std::string candidate;
            if (starts_with(logical_name, "text_encoder.")) {
                ++clip_l_parameters;
                candidate = "conditioner.embedders.0.transformer." +
                    logical_name.substr(std::string("text_encoder.").size());
            } else if (starts_with(logical_name, "text_encoder_2.")) {
                ++openclip_parameters;
                candidate = "conditioner.embedders.1.model.transformer." +
                    logical_name.substr(std::string("text_encoder_2.").size());
            } else {
                continue;
            }
            if (!keys.contains(candidate)) {
                throw std::runtime_error(
                    "reference checkpoint key is missing for " + logical_name +
                    ": expected " + candidate);
            }
        }

        require(clip_l_parameters == 196, "unexpected CLIP-L parameter count");
        require(openclip_parameters == 517, "unexpected OpenCLIP parameter count");

        std::size_t scales = 0;
        std::size_t quant_metadata = 0;
        for (const auto& key : keys) {
            if (key.ends_with(".weight_scale")) ++scales;
            if (key.ends_with(".comfy_quant")) ++quant_metadata;
        }
        require(scales == 1008, "unexpected weight_scale count");
        require(quant_metadata == 1008, "unexpected comfy_quant count");

        std::cout << "INT8 reference key layout test passed: "
                  << clip_l_parameters << " CLIP-L + "
                  << openclip_parameters << " OpenCLIP parameters, "
                  << scales << " quantized Linear scale tensors\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "INT8 reference key layout test failed: " << error.what() << '\n';
        return 1;
    }
}
