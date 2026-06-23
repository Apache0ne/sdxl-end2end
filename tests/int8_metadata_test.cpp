#include "sdxl/sdxl.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct TensorRecord {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> data;
};

[[nodiscard]] std::string shape_json(const std::vector<std::uint64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) out << ',';
        out << shape[index];
    }
    out << ']';
    return out.str();
}

void write_safetensors(const std::filesystem::path& path,
                       const std::vector<TensorRecord>& records) {
    std::ostringstream header;
    header << '{';
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < records.size(); ++index) {
        const TensorRecord& record = records[index];
        if (index != 0) header << ',';
        header << '\"' << record.name << "\":{";
        header << "\"dtype\":\"" << record.dtype << "\",";
        header << "\"shape\":" << shape_json(record.shape) << ',';
        header << "\"data_offsets\":[" << offset << ','
               << offset + record.data.size() << "]}";
        offset += record.data.size();
    }
    header << '}';
    std::string header_text = header.str();
    while (header_text.size() % 8 != 0) header_text.push_back(' ');

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create temporary safetensors file");
    const std::uint64_t header_size = header_text.size();
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const TensorRecord& record : records) {
        output.write(reinterpret_cast<const char*>(record.data.data()),
                     static_cast<std::streamsize>(record.data.size()));
    }
    if (!output) throw std::runtime_error("failed writing temporary safetensors file");
}

[[nodiscard]] std::vector<std::byte> repeated_bytes(std::size_t count, std::uint8_t value) {
    return std::vector<std::byte>(count, static_cast<std::byte>(value));
}

[[nodiscard]] std::vector<std::byte> float_bytes(std::size_t count, float value) {
    std::vector<float> values(count, value);
    const auto* begin = reinterpret_cast<const std::byte*>(values.data());
    return {begin, begin + values.size() * sizeof(float)};
}

[[nodiscard]] std::vector<std::byte> string_bytes(std::string_view value) {
    const auto* begin = reinterpret_cast<const std::byte*>(value.data());
    return {begin, begin + value.size()};
}

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

} // namespace

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "sdxl_int8_metadata_test.safetensors";
    try {
        constexpr std::size_t time_rows = 1280;
        constexpr std::size_t time_columns = 320;
        constexpr std::size_t qkv_rows = 3840;
        constexpr std::size_t qkv_columns = 1280;
        const std::string config =
            R"({"convrot":true,"convrot_groupsize":256,"per_row":true})";

        write_safetensors(path, {
            {"model.diffusion_model.time_embed.0.weight", "I8",
             {time_rows, time_columns}, repeated_bytes(time_rows * time_columns, 1)},
            {"model.diffusion_model.time_embed.0.weight_scale", "F32",
             {time_rows, 1}, float_bytes(time_rows, 0.01F)},
            {"model.diffusion_model.time_embed.0.comfy_quant", "U8",
             {config.size()}, string_bytes(config)},
            {"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight", "I8",
             {qkv_rows, qkv_columns}, repeated_bytes(qkv_rows * qkv_columns, 2)},
            {"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight_scale", "F32",
             {qkv_rows, 1}, float_bytes(qkv_rows, 0.02F)},
            {"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_comfy_quant", "U8",
             {config.size()}, string_bytes(config)},
        });

        sdxl::SDXLModel model;
        sdxl::LoadOptions options;
        options.strict = false;
        const sdxl::LoadResult result = sdxl::SDXLWeightLoader(options).load(model, path);
        require(result.parameters_bound >= 4, "expected INT8 parameters were not bound");

        const sdxl::ParameterSlot* time =
            model.graph().find_parameter("unet.time_embedding.linear_1.weight");
        require(time != nullptr && time->tensor.has_value(), "UNet INT8 weight was not bound");
        require(time->tensor->dtype == sdxl::DType::I8, "UNet INT8 dtype was not preserved");
        require(time->quantization.has_value(), "UNet quantization metadata missing");
        require(time->quantization->weight_scale.has_value(), "UNet weight scale missing");
        require(time->quantization->weight_scale->element_count() == time_rows,
                "UNet scale shape mismatch");
        require(time->quantization->convrot, "UNet ConvRot flag missing");
        require(time->quantization->per_row, "UNet per-row flag missing");
        require(time->quantization->convrot_group_size == 256,
                "UNet ConvRot group size mismatch");

        for (const std::string projection : {"q_proj", "k_proj", "v_proj"}) {
            const std::string name =
                "text_encoder_2.text_model.encoder.layers.0.self_attn." + projection + ".weight";
            const sdxl::ParameterSlot* slot = model.graph().find_parameter(name);
            require(slot != nullptr && slot->tensor.has_value(), "OpenCLIP QKV slice missing");
            require(slot->tensor->dtype == sdxl::DType::I8, "OpenCLIP QKV dtype mismatch");
            require(slot->quantization.has_value() &&
                    slot->quantization->weight_scale.has_value(),
                    "OpenCLIP QKV quantization scale missing");
            require(slot->quantization->weight_scale->element_count() == 1280,
                    "OpenCLIP QKV scale was not sliced with its weight");
            require(slot->quantization->convrot &&
                    slot->quantization->convrot_group_size == 256,
                    "OpenCLIP QKV ConvRot metadata mismatch");
        }

        std::filesystem::remove(path);
        std::cout << "INT8 checkpoint metadata test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << "INT8 checkpoint metadata test failed: " << error.what() << '\n';
        return 1;
    }
}
