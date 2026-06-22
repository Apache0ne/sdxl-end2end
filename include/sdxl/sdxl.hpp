#pragma once

#include "sdxl/model.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace sdxl {

struct SDXLConfig {
    struct TextEncoder {
        std::uint64_t vocab_size = 0;
        std::uint64_t max_positions = 77;
        std::uint64_t hidden_size = 0;
        std::uint64_t intermediate_size = 0;
        std::uint64_t layers = 0;
        std::uint64_t heads = 0;
        std::uint64_t projection_dim = 0;
        bool with_projection = false;
        std::string activation;
    };

    TextEncoder clip_l;
    TextEncoder openclip_big_g;

    std::vector<std::uint64_t> unet_channels;
    std::vector<std::uint64_t> unet_heads;
    std::vector<std::uint64_t> unet_transformer_depth;
    std::uint64_t unet_cross_attention_dim = 2048;
    std::uint64_t unet_time_embed_dim = 1280;
    std::uint64_t unet_addition_time_embed_dim = 256;
    std::uint64_t unet_addition_input_dim = 2816;
    std::uint64_t unet_layers_per_block = 2;
    std::uint64_t unet_norm_groups = 32;

    std::vector<std::uint64_t> vae_channels;
    std::uint64_t vae_latent_channels = 4;
    std::uint64_t vae_layers_per_block = 2;
    std::uint64_t vae_norm_groups = 32;
    float vae_scaling_factor = 0.13025F;

    [[nodiscard]] static SDXLConfig base_1_0();
};

class SDXLModel final {
public:
    explicit SDXLModel(SDXLConfig config = SDXLConfig::base_1_0());

    [[nodiscard]] ModelGraph& graph() noexcept { return graph_; }
    [[nodiscard]] const ModelGraph& graph() const noexcept { return graph_; }
    [[nodiscard]] const SDXLConfig& config() const noexcept { return config_; }

private:
    SDXLConfig config_;
    ModelGraph graph_;
};

enum class CheckpointLayout {
    DiffusersDirectory,
    OriginalSingleFile,
    Unknown
};

struct LoadOptions {
    bool strict = true;
    bool prefer_fp16_variant = true;
    bool allow_unexpected_tensors = true;
};

struct LoadResult {
    CheckpointLayout layout = CheckpointLayout::Unknown;
    std::size_t files_mapped = 0;
    std::size_t tensors_discovered = 0;
    std::size_t parameters_bound = 0;
    ValidationReport validation;
    std::vector<std::string> notes;
};

class SDXLWeightLoader final {
public:
    explicit SDXLWeightLoader(LoadOptions options = {});

    [[nodiscard]] LoadResult load(SDXLModel& model, const std::filesystem::path& path);

private:
    LoadOptions options_;
};

} // namespace sdxl
