#include "sdxl/sdxl.hpp"
#include "sdxl/json.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_set>

namespace sdxl {

namespace {

using Shape = std::vector<std::uint64_t>;

void add_linear(ModuleNode& parent, const std::string& name,
                std::uint64_t in_features, std::uint64_t out_features,
                bool bias = true) {
    ModuleNode& node = parent.child(name, "Linear");
    node.attribute("in_features", std::to_string(in_features));
    node.attribute("out_features", std::to_string(out_features));
    node.parameter("weight", {out_features, in_features});
    if (bias) node.parameter("bias", {out_features});
}

void add_conv(ModuleNode& parent, const std::string& name,
              std::uint64_t in_channels, std::uint64_t out_channels,
              std::uint64_t kernel, bool bias = true) {
    ModuleNode& node = parent.child(name, "Conv2D");
    node.attribute("in_channels", std::to_string(in_channels));
    node.attribute("out_channels", std::to_string(out_channels));
    node.attribute("kernel", std::to_string(kernel));
    node.parameter("weight", {out_channels, in_channels, kernel, kernel});
    if (bias) node.parameter("bias", {out_channels});
}

void add_norm(ModuleNode& parent, const std::string& name,
              std::uint64_t channels, const std::string& type) {
    ModuleNode& node = parent.child(name, type);
    node.parameter("weight", {channels});
    node.parameter("bias", {channels});
}

void add_embedding(ModuleNode& parent, const std::string& name,
                   std::uint64_t count, std::uint64_t width) {
    ModuleNode& node = parent.child(name, "Embedding");
    node.parameter("weight", {count, width});
}

void add_clip_text_encoder(ModuleNode& root,
                           const std::string& component_name,
                           const SDXLConfig::TextEncoder& cfg) {
    ModuleNode& encoder = root.child(component_name,
        cfg.with_projection ? "CLIPTextModelWithProjection" : "CLIPTextModel");
    encoder.attribute("hidden_size", std::to_string(cfg.hidden_size));
    encoder.attribute("intermediate_size", std::to_string(cfg.intermediate_size));
    encoder.attribute("num_hidden_layers", std::to_string(cfg.layers));
    encoder.attribute("num_attention_heads", std::to_string(cfg.heads));
    encoder.attribute("activation", cfg.activation);

    add_embedding(encoder, "text_model.embeddings.token_embedding", cfg.vocab_size, cfg.hidden_size);
    add_embedding(encoder, "text_model.embeddings.position_embedding", cfg.max_positions, cfg.hidden_size);

    for (std::uint64_t i = 0; i < cfg.layers; ++i) {
        const std::string p = "text_model.encoder.layers." + std::to_string(i);
        add_norm(encoder, p + ".layer_norm1", cfg.hidden_size, "LayerNorm");
        add_linear(encoder, p + ".self_attn.q_proj", cfg.hidden_size, cfg.hidden_size, true);
        add_linear(encoder, p + ".self_attn.k_proj", cfg.hidden_size, cfg.hidden_size, true);
        add_linear(encoder, p + ".self_attn.v_proj", cfg.hidden_size, cfg.hidden_size, true);
        add_linear(encoder, p + ".self_attn.out_proj", cfg.hidden_size, cfg.hidden_size, true);
        add_norm(encoder, p + ".layer_norm2", cfg.hidden_size, "LayerNorm");
        add_linear(encoder, p + ".mlp.fc1", cfg.hidden_size, cfg.intermediate_size, true);
        add_linear(encoder, p + ".mlp.fc2", cfg.intermediate_size, cfg.hidden_size, true);
    }

    add_norm(encoder, "text_model.final_layer_norm", cfg.hidden_size, "LayerNorm");
    if (cfg.with_projection) {
        add_linear(encoder, "text_projection", cfg.hidden_size, cfg.projection_dim, false);
    }
}

void add_resnet_2d(ModuleNode& parent,
                   const std::string& prefix,
                   std::uint64_t in_channels,
                   std::uint64_t out_channels,
                   std::uint64_t time_embed_dim,
                   bool has_time_embedding) {
    ModuleNode& node = parent.child(prefix, "ResnetBlock2D");
    node.attribute("in_channels", std::to_string(in_channels));
    node.attribute("out_channels", std::to_string(out_channels));
    node.parameter("norm1.weight", {in_channels});
    node.parameter("norm1.bias", {in_channels});
    node.parameter("conv1.weight", {out_channels, in_channels, 3, 3});
    node.parameter("conv1.bias", {out_channels});
    if (has_time_embedding) {
        node.parameter("time_emb_proj.weight", {out_channels, time_embed_dim});
        node.parameter("time_emb_proj.bias", {out_channels});
    }
    node.parameter("norm2.weight", {out_channels});
    node.parameter("norm2.bias", {out_channels});
    node.parameter("conv2.weight", {out_channels, out_channels, 3, 3});
    node.parameter("conv2.bias", {out_channels});
    if (in_channels != out_channels) {
        node.parameter("conv_shortcut.weight", {out_channels, in_channels, 1, 1});
        node.parameter("conv_shortcut.bias", {out_channels});
    }
}

void add_transformer_2d(ModuleNode& parent,
                        const std::string& prefix,
                        std::uint64_t channels,
                        std::uint64_t cross_attention_dim,
                        std::uint64_t depth,
                        std::uint64_t heads) {
    ModuleNode& node = parent.child(prefix, "Transformer2DModel");
    node.attribute("channels", std::to_string(channels));
    node.attribute("cross_attention_dim", std::to_string(cross_attention_dim));
    node.attribute("depth", std::to_string(depth));
    node.attribute("num_attention_heads", std::to_string(heads));
    node.attribute("attention_head_dim", std::to_string(channels / heads));

    node.parameter("norm.weight", {channels});
    node.parameter("norm.bias", {channels});
    node.parameter("proj_in.weight", {channels, channels});
    node.parameter("proj_in.bias", {channels});

    for (std::uint64_t i = 0; i < depth; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i);
        node.parameter(p + ".norm1.weight", {channels});
        node.parameter(p + ".norm1.bias", {channels});
        node.parameter(p + ".attn1.to_q.weight", {channels, channels});
        node.parameter(p + ".attn1.to_k.weight", {channels, channels});
        node.parameter(p + ".attn1.to_v.weight", {channels, channels});
        node.parameter(p + ".attn1.to_out.0.weight", {channels, channels});
        node.parameter(p + ".attn1.to_out.0.bias", {channels});

        node.parameter(p + ".norm2.weight", {channels});
        node.parameter(p + ".norm2.bias", {channels});
        node.parameter(p + ".attn2.to_q.weight", {channels, channels});
        node.parameter(p + ".attn2.to_k.weight", {channels, cross_attention_dim});
        node.parameter(p + ".attn2.to_v.weight", {channels, cross_attention_dim});
        node.parameter(p + ".attn2.to_out.0.weight", {channels, channels});
        node.parameter(p + ".attn2.to_out.0.bias", {channels});

        node.parameter(p + ".norm3.weight", {channels});
        node.parameter(p + ".norm3.bias", {channels});
        node.parameter(p + ".ff.net.0.proj.weight", {channels * 8, channels});
        node.parameter(p + ".ff.net.0.proj.bias", {channels * 8});
        node.parameter(p + ".ff.net.2.weight", {channels, channels * 4});
        node.parameter(p + ".ff.net.2.bias", {channels});
    }

    node.parameter("proj_out.weight", {channels, channels});
    node.parameter("proj_out.bias", {channels});
}

void add_unet(ModuleNode& root, const SDXLConfig& cfg) {
    ModuleNode& unet = root.child("unet", "UNet2DConditionModel");
    unet.attribute("in_channels", "4");
    unet.attribute("out_channels", "4");
    unet.attribute("cross_attention_dim", std::to_string(cfg.unet_cross_attention_dim));
    unet.attribute("addition_embed_type", "text_time");
    unet.attribute("addition_time_embed_dim", std::to_string(cfg.unet_addition_time_embed_dim));
    unet.attribute("use_linear_projection", "true");

    add_conv(unet, "conv_in", 4, cfg.unet_channels[0], 3, true);
    add_linear(unet, "time_embedding.linear_1", cfg.unet_channels[0], cfg.unet_time_embed_dim, true);
    add_linear(unet, "time_embedding.linear_2", cfg.unet_time_embed_dim, cfg.unet_time_embed_dim, true);
    add_linear(unet, "add_embedding.linear_1", cfg.unet_addition_input_dim, cfg.unet_time_embed_dim, true);
    add_linear(unet, "add_embedding.linear_2", cfg.unet_time_embed_dim, cfg.unet_time_embed_dim, true);

    std::vector<std::uint64_t> residual_channels;
    residual_channels.push_back(cfg.unet_channels[0]);
    std::uint64_t previous = cfg.unet_channels[0];
    for (std::size_t block = 0; block < cfg.unet_channels.size(); ++block) {
        const std::uint64_t out = cfg.unet_channels[block];
        for (std::uint64_t layer = 0; layer < cfg.unet_layers_per_block; ++layer) {
            const std::uint64_t in = layer == 0 ? previous : out;
            add_resnet_2d(unet,
                "down_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer),
                in, out, cfg.unet_time_embed_dim, true);
            if (block > 0) {
                add_transformer_2d(unet,
                    "down_blocks." + std::to_string(block) + ".attentions." + std::to_string(layer),
                    out, cfg.unet_cross_attention_dim, cfg.unet_transformer_depth[block],
                    cfg.unet_heads[block]);
            }
            previous = out;
            residual_channels.push_back(out);
        }
        if (block + 1 < cfg.unet_channels.size()) {
            add_conv(unet,
                     "down_blocks." + std::to_string(block) + ".downsamplers.0.conv",
                     out, out, 3, true);
            residual_channels.push_back(out);
        }
    }

    const std::uint64_t mid = cfg.unet_channels.back();
    add_resnet_2d(unet, "mid_block.resnets.0", mid, mid, cfg.unet_time_embed_dim, true);
    add_transformer_2d(unet, "mid_block.attentions.0", mid,
                       cfg.unet_cross_attention_dim, cfg.unet_transformer_depth.back(),
                       cfg.unet_heads.back());
    add_resnet_2d(unet, "mid_block.resnets.1", mid, mid, cfg.unet_time_embed_dim, true);

    std::uint64_t current_channels = cfg.unet_channels.back();
    const std::size_t up_layer_count = static_cast<std::size_t>(cfg.unet_layers_per_block + 1);
    for (std::size_t block = 0; block < cfg.unet_channels.size(); ++block) {
        const std::size_t down_index = cfg.unet_channels.size() - 1 - block;
        const std::uint64_t out_channels = cfg.unet_channels[down_index];
        for (std::size_t layer = 0; layer < up_layer_count; ++layer) {
            if (residual_channels.empty()) throw Error("invalid UNet residual-channel construction");
            const std::uint64_t skip_channels = residual_channels.back();
            residual_channels.pop_back();
            add_resnet_2d(unet,
                "up_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer),
                current_channels + skip_channels, out_channels, cfg.unet_time_embed_dim, true);
            current_channels = out_channels;
            if (down_index > 0) {
                add_transformer_2d(unet,
                    "up_blocks." + std::to_string(block) + ".attentions." + std::to_string(layer),
                    out_channels, cfg.unet_cross_attention_dim,
                    cfg.unet_transformer_depth[down_index], cfg.unet_heads[down_index]);
            }
        }
        if (block + 1 < cfg.unet_channels.size()) {
            add_conv(unet,
                     "up_blocks." + std::to_string(block) + ".upsamplers.0.conv",
                     out_channels, out_channels, 3, true);
        }
    }
    if (!residual_channels.empty()) throw Error("UNet residual-channel construction left unused skips");

    add_norm(unet, "conv_norm_out", cfg.unet_channels[0], "GroupNorm");
    add_conv(unet, "conv_out", cfg.unet_channels[0], 4, 3, true);
}

void add_vae_attention(ModuleNode& parent, const std::string& prefix, std::uint64_t channels) {
    ModuleNode& node = parent.child(prefix, "Attention");
    node.parameter("group_norm.weight", {channels});
    node.parameter("group_norm.bias", {channels});
    node.parameter("to_q.weight", {channels, channels});
    node.parameter("to_q.bias", {channels});
    node.parameter("to_k.weight", {channels, channels});
    node.parameter("to_k.bias", {channels});
    node.parameter("to_v.weight", {channels, channels});
    node.parameter("to_v.bias", {channels});
    node.parameter("to_out.0.weight", {channels, channels});
    node.parameter("to_out.0.bias", {channels});
}

void add_vae(ModuleNode& root, const SDXLConfig& cfg) {
    ModuleNode& vae = root.child("vae", "AutoencoderKL");
    vae.attribute("latent_channels", std::to_string(cfg.vae_latent_channels));
    vae.attribute("scaling_factor", std::to_string(cfg.vae_scaling_factor));
    vae.attribute("force_upcast", "true");

    add_conv(vae, "encoder.conv_in", 3, cfg.vae_channels[0], 3, true);
    std::uint64_t in = cfg.vae_channels[0];
    for (std::size_t block = 0; block < cfg.vae_channels.size(); ++block) {
        const std::uint64_t out = cfg.vae_channels[block];
        for (std::uint64_t layer = 0; layer < cfg.vae_layers_per_block; ++layer) {
            add_resnet_2d(vae,
                "encoder.down_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer),
                layer == 0 ? in : out, out, 0, false);
        }
        if (block + 1 < cfg.vae_channels.size()) {
            add_conv(vae,
                     "encoder.down_blocks." + std::to_string(block) + ".downsamplers.0.conv",
                     out, out, 3, true);
        }
        in = out;
    }

    const std::uint64_t mid = cfg.vae_channels.back();
    add_resnet_2d(vae, "encoder.mid_block.resnets.0", mid, mid, 0, false);
    add_vae_attention(vae, "encoder.mid_block.attentions.0", mid);
    add_resnet_2d(vae, "encoder.mid_block.resnets.1", mid, mid, 0, false);
    add_norm(vae, "encoder.conv_norm_out", mid, "GroupNorm");
    add_conv(vae, "encoder.conv_out", mid, cfg.vae_latent_channels * 2, 3, true);
    add_conv(vae, "quant_conv", cfg.vae_latent_channels * 2, cfg.vae_latent_channels * 2, 1, true);

    add_conv(vae, "post_quant_conv", cfg.vae_latent_channels, cfg.vae_latent_channels, 1, true);
    add_conv(vae, "decoder.conv_in", cfg.vae_latent_channels, mid, 3, true);
    add_resnet_2d(vae, "decoder.mid_block.resnets.0", mid, mid, 0, false);
    add_vae_attention(vae, "decoder.mid_block.attentions.0", mid);
    add_resnet_2d(vae, "decoder.mid_block.resnets.1", mid, mid, 0, false);

    const std::array<std::uint64_t, 4> decoder_out{{512, 512, 256, 128}};
    std::uint64_t decoder_in = 512;
    for (std::size_t block = 0; block < decoder_out.size(); ++block) {
        const std::uint64_t out = decoder_out[block];
        for (std::uint64_t layer = 0; layer < cfg.vae_layers_per_block + 1; ++layer) {
            add_resnet_2d(vae,
                "decoder.up_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer),
                layer == 0 ? decoder_in : out, out, 0, false);
        }
        if (block + 1 < decoder_out.size()) {
            add_conv(vae,
                     "decoder.up_blocks." + std::to_string(block) + ".upsamplers.0.conv",
                     out, out, 3, true);
        }
        decoder_in = out;
    }
    add_norm(vae, "decoder.conv_norm_out", 128, "GroupNorm");
    add_conv(vae, "decoder.conv_out", 128, 3, 3, true);
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix) noexcept {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool ends_with(std::string_view value, std::string_view suffix) noexcept {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::string strip_prefix(std::string_view value, std::string_view prefix) {
    if (!starts_with(value, prefix)) return {};
    return std::string(value.substr(prefix.size()));
}

[[nodiscard]] std::string replace_first(std::string value,
                                        std::string_view from,
                                        std::string_view to) {
    const auto pos = value.find(from);
    if (pos != std::string::npos) value.replace(pos, from.size(), to);
    return value;
}

[[nodiscard]] std::string map_resnet_suffix_to_ldm(std::string suffix) {
    if (starts_with(suffix, "norm1.")) return replace_first(std::move(suffix), "norm1.", "in_layers.0.");
    if (starts_with(suffix, "conv1.")) return replace_first(std::move(suffix), "conv1.", "in_layers.2.");
    if (starts_with(suffix, "time_emb_proj.")) return replace_first(std::move(suffix), "time_emb_proj.", "emb_layers.1.");
    if (starts_with(suffix, "norm2.")) return replace_first(std::move(suffix), "norm2.", "out_layers.0.");
    if (starts_with(suffix, "conv2.")) return replace_first(std::move(suffix), "conv2.", "out_layers.3.");
    if (starts_with(suffix, "conv_shortcut.")) return replace_first(std::move(suffix), "conv_shortcut.", "skip_connection.");
    return suffix;
}

[[nodiscard]] std::vector<std::string> unet_original_candidates(std::string_view logical) {
    const std::string key = strip_prefix(logical, "unet.");
    if (key.empty()) return {};
    const std::string base = "model.diffusion_model.";

    const std::map<std::string, std::string> direct{
        {"time_embedding.linear_1.weight", "time_embed.0.weight"},
        {"time_embedding.linear_1.bias", "time_embed.0.bias"},
        {"time_embedding.linear_2.weight", "time_embed.2.weight"},
        {"time_embedding.linear_2.bias", "time_embed.2.bias"},
        {"conv_in.weight", "input_blocks.0.0.weight"},
        {"conv_in.bias", "input_blocks.0.0.bias"},
        {"conv_norm_out.weight", "out.0.weight"},
        {"conv_norm_out.bias", "out.0.bias"},
        {"conv_out.weight", "out.2.weight"},
        {"conv_out.bias", "out.2.bias"},
        {"add_embedding.linear_1.weight", "label_emb.0.0.weight"},
        {"add_embedding.linear_1.bias", "label_emb.0.0.bias"},
        {"add_embedding.linear_2.weight", "label_emb.0.2.weight"},
        {"add_embedding.linear_2.bias", "label_emb.0.2.bias"},
    };
    if (const auto it = direct.find(key); it != direct.end()) return {base + it->second};

    std::smatch match;
    const std::string s = key;
    if (std::regex_match(s, match, std::regex(R"(^down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        const int layer = std::stoi(match[2].str());
        return {base + "input_blocks." + std::to_string(3 * block + layer + 1) + ".0." +
                map_resnet_suffix_to_ldm(match[3].str())};
    }
    if (std::regex_match(s, match, std::regex(R"(^down_blocks\.(\d+)\.attentions\.(\d+)\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        const int layer = std::stoi(match[2].str());
        return {base + "input_blocks." + std::to_string(3 * block + layer + 1) + ".1." + match[3].str()};
    }
    if (std::regex_match(s, match, std::regex(R"(^down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        return {base + "input_blocks." + std::to_string(3 * (block + 1)) + ".0.op." + match[2].str()};
    }
    if (std::regex_match(s, match, std::regex(R"(^mid_block\.resnets\.(\d+)\.(.+)$)"))) {
        const int layer = std::stoi(match[1].str());
        return {base + "middle_block." + std::to_string(layer == 0 ? 0 : 2) + "." +
                map_resnet_suffix_to_ldm(match[2].str())};
    }
    if (std::regex_match(s, match, std::regex(R"(^mid_block\.attentions\.0\.(.+)$)"))) {
        return {base + "middle_block.1." + match[1].str()};
    }
    if (std::regex_match(s, match, std::regex(R"(^up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        const int layer = std::stoi(match[2].str());
        return {base + "output_blocks." + std::to_string(3 * block + layer) + ".0." +
                map_resnet_suffix_to_ldm(match[3].str())};
    }
    if (std::regex_match(s, match, std::regex(R"(^up_blocks\.(\d+)\.attentions\.(\d+)\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        const int layer = std::stoi(match[2].str());
        return {base + "output_blocks." + std::to_string(3 * block + layer) + ".1." + match[3].str()};
    }
    if (std::regex_match(s, match, std::regex(R"(^up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$)"))) {
        const int block = std::stoi(match[1].str());
        const std::string output = base + "output_blocks." + std::to_string(3 * block + 2) + ".";
        return {output + "2.conv." + match[2].str(), output + "1.conv." + match[2].str()};
    }
    return {};
}

[[nodiscard]] std::string map_vae_resnet_suffix_to_ldm(std::string suffix) {
    if (starts_with(suffix, "conv_shortcut.")) return replace_first(std::move(suffix), "conv_shortcut.", "nin_shortcut.");
    return suffix;
}

[[nodiscard]] std::string map_vae_attention_suffix_to_ldm(std::string suffix) {
    if (starts_with(suffix, "group_norm.")) return replace_first(std::move(suffix), "group_norm.", "norm.");
    if (starts_with(suffix, "to_q.")) return replace_first(std::move(suffix), "to_q.", "q.");
    if (starts_with(suffix, "to_k.")) return replace_first(std::move(suffix), "to_k.", "k.");
    if (starts_with(suffix, "to_v.")) return replace_first(std::move(suffix), "to_v.", "v.");
    if (starts_with(suffix, "to_out.0.")) return replace_first(std::move(suffix), "to_out.0.", "proj_out.");
    return suffix;
}

[[nodiscard]] std::vector<std::string> vae_original_candidates(std::string_view logical) {
    const std::string key = strip_prefix(logical, "vae.");
    if (key.empty()) return {};
    const std::string base = "first_stage_model.";
    std::smatch match;

    if (std::regex_match(key, match, std::regex(R"(^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$)"))) {
        return {base + "encoder.down." + match[1].str() + ".block." + match[2].str() + "." +
                map_vae_resnet_suffix_to_ldm(match[3].str())};
    }
    if (std::regex_match(key, match, std::regex(R"(^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$)"))) {
        return {base + "encoder.down." + match[1].str() + ".downsample.conv." + match[2].str()};
    }
    if (std::regex_match(key, match, std::regex(R"(^encoder\.mid_block\.resnets\.(\d+)\.(.+)$)"))) {
        return {base + "encoder.mid.block_" + std::to_string(std::stoi(match[1].str()) + 1) + "." +
                map_vae_resnet_suffix_to_ldm(match[2].str())};
    }
    if (std::regex_match(key, match, std::regex(R"(^encoder\.mid_block\.attentions\.0\.(.+)$)"))) {
        return {base + "encoder.mid.attn_1." + map_vae_attention_suffix_to_ldm(match[1].str())};
    }
    if (std::regex_match(key, match, std::regex(R"(^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$)"))) {
        const int diffusers_block = std::stoi(match[1].str());
        const int original_block = 3 - diffusers_block;
        return {base + "decoder.up." + std::to_string(original_block) + ".block." + match[2].str() + "." +
                map_vae_resnet_suffix_to_ldm(match[3].str())};
    }
    if (std::regex_match(key, match, std::regex(R"(^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$)"))) {
        const int original_block = 3 - std::stoi(match[1].str());
        return {base + "decoder.up." + std::to_string(original_block) + ".upsample.conv." + match[2].str()};
    }
    if (std::regex_match(key, match, std::regex(R"(^decoder\.mid_block\.resnets\.(\d+)\.(.+)$)"))) {
        return {base + "decoder.mid.block_" + std::to_string(std::stoi(match[1].str()) + 1) + "." +
                map_vae_resnet_suffix_to_ldm(match[2].str())};
    }
    if (std::regex_match(key, match, std::regex(R"(^decoder\.mid_block\.attentions\.0\.(.+)$)"))) {
        return {base + "decoder.mid.attn_1." + map_vae_attention_suffix_to_ldm(match[1].str())};
    }

    std::string direct = key;
    direct = replace_first(std::move(direct), "encoder.conv_norm_out.", "encoder.norm_out.");
    direct = replace_first(std::move(direct), "decoder.conv_norm_out.", "decoder.norm_out.");
    return {base + direct};
}

struct DerivedLookup {
    std::string key;
    enum class Transform { None, SqueezeTrailingOnes, Transpose01, QkvSlice0, QkvSlice1, QkvSlice2 } transform = Transform::None;
};

[[nodiscard]] std::vector<DerivedLookup> text_encoder_2_original_candidates(std::string_view logical) {
    const std::string key = strip_prefix(logical, "text_encoder_2.");
    if (key.empty()) return {};
    const std::array<std::string, 2> bases{{
        "conditioner.embedders.1.model.",
        "conditioner.embedders.1.transformer."
    }};
    std::vector<DerivedLookup> out;
    auto add_all = [&](const std::string& suffix, DerivedLookup::Transform transform = DerivedLookup::Transform::None) {
        for (const auto& base : bases) out.push_back({base + suffix, transform});
    };

    if (key == "text_model.embeddings.token_embedding.weight") {
        add_all("token_embedding.weight");
        return out;
    }
    if (key == "text_model.embeddings.position_embedding.weight") {
        add_all("positional_embedding");
        return out;
    }
    if (key == "text_model.final_layer_norm.weight") {
        add_all("ln_final.weight");
        return out;
    }
    if (key == "text_model.final_layer_norm.bias") {
        add_all("ln_final.bias");
        return out;
    }
    if (key == "text_projection.weight") {
        add_all("text_projection", DerivedLookup::Transform::Transpose01);
        add_all("text_projection.weight");
        return out;
    }

    std::smatch match;
    if (!std::regex_match(key, match, std::regex(R"(^text_model\.encoder\.layers\.(\d+)\.(.+)$)"))) {
        return out;
    }
    const std::string layer = "transformer.resblocks." + match[1].str() + ".";
    const std::string suffix = match[2].str();
    if (suffix == "layer_norm1.weight") add_all(layer + "ln_1.weight");
    else if (suffix == "layer_norm1.bias") add_all(layer + "ln_1.bias");
    else if (suffix == "layer_norm2.weight") add_all(layer + "ln_2.weight");
    else if (suffix == "layer_norm2.bias") add_all(layer + "ln_2.bias");
    else if (suffix == "self_attn.out_proj.weight") add_all(layer + "attn.out_proj.weight");
    else if (suffix == "self_attn.out_proj.bias") add_all(layer + "attn.out_proj.bias");
    else if (suffix == "self_attn.q_proj.weight") add_all(layer + "attn.in_proj_weight", DerivedLookup::Transform::QkvSlice0);
    else if (suffix == "self_attn.k_proj.weight") add_all(layer + "attn.in_proj_weight", DerivedLookup::Transform::QkvSlice1);
    else if (suffix == "self_attn.v_proj.weight") add_all(layer + "attn.in_proj_weight", DerivedLookup::Transform::QkvSlice2);
    else if (suffix == "self_attn.q_proj.bias") add_all(layer + "attn.in_proj_bias", DerivedLookup::Transform::QkvSlice0);
    else if (suffix == "self_attn.k_proj.bias") add_all(layer + "attn.in_proj_bias", DerivedLookup::Transform::QkvSlice1);
    else if (suffix == "self_attn.v_proj.bias") add_all(layer + "attn.in_proj_bias", DerivedLookup::Transform::QkvSlice2);
    else if (suffix == "mlp.fc1.weight") add_all(layer + "mlp.c_fc.weight");
    else if (suffix == "mlp.fc1.bias") add_all(layer + "mlp.c_fc.bias");
    else if (suffix == "mlp.fc2.weight") add_all(layer + "mlp.c_proj.weight");
    else if (suffix == "mlp.fc2.bias") add_all(layer + "mlp.c_proj.bias");
    return out;
}

[[nodiscard]] TensorView apply_transform(TensorView view,
                                         DerivedLookup::Transform transform,
                                         const Shape& expected) {
    switch (transform) {
    case DerivedLookup::Transform::None:
        break;
    case DerivedLookup::Transform::SqueezeTrailingOnes:
        view = view.squeeze_trailing_ones();
        break;
    case DerivedLookup::Transform::Transpose01:
        view = view.transpose(0, 1);
        break;
    case DerivedLookup::Transform::QkvSlice0:
    case DerivedLookup::Transform::QkvSlice1:
    case DerivedLookup::Transform::QkvSlice2: {
        if (expected.empty()) throw Error("cannot derive QKV slice for scalar tensor");
        const std::uint64_t width = expected[0];
        const std::uint64_t index = transform == DerivedLookup::Transform::QkvSlice0 ? 0 :
                                    transform == DerivedLookup::Transform::QkvSlice1 ? 1 : 2;
        view = view.slice(0, index * width, width);
        break;
    }
    }
    if (view.shape != expected) {
        TensorView squeezed = view.squeeze_trailing_ones();
        if (squeezed.shape == expected) view = std::move(squeezed);
    }
    return view;
}

[[nodiscard]] std::optional<TensorView> unique_suffix_lookup(const WeightStore& store,
                                                             std::string_view suffix) {
    const TensorView* match = nullptr;
    for (const auto& [key, view] : store.tensors()) {
        if (!ends_with(key, suffix)) continue;
        if (match != nullptr) return std::nullopt;
        match = &view;
    }
    if (match == nullptr) return std::nullopt;
    return *match;
}

[[nodiscard]] std::string strip_view_annotation(std::string value) {
    const auto bracket = value.find('[');
    if (bracket != std::string::npos) value.resize(bracket);
    return value;
}

[[nodiscard]] std::vector<std::string> quantization_prefix_candidates(
    const ParameterSlot& slot, const TensorView& bound) {
    std::vector<std::string> prefixes;
    auto insert = [&](std::string prefix) {
        if (prefix.empty()) return;
        if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) {
            prefixes.push_back(std::move(prefix));
        }
    };
    auto add_key = [&](std::string key) {
        key = strip_view_annotation(std::move(key));
        constexpr std::string_view dot_weight = ".weight";
        constexpr std::string_view underscore_weight = "_weight";
        if (ends_with(key, dot_weight)) {
            // PyTorch module layout: <module>.weight -> <module>.weight_scale.
            insert(key.substr(0, key.size() - dot_weight.size() + 1));
        } else if (ends_with(key, underscore_weight)) {
            // Original OpenCLIP fused QKV layout commonly uses
            // attn.in_proj_weight + attn.in_proj_weight_scale. Keep the
            // append-style spelling too for converters that serialize
            // attn.in_proj_weight.weight_scale.
            insert(key.substr(0, key.size() - std::string_view("weight").size()));
            insert(key + '.');
        } else {
            // Parameter names such as OpenCLIP text_projection have no final
            // ".weight" in original SDXL checkpoints.
            insert(key + '.');
        }
    };

    add_key(bound.source_key);
    add_key(slot.logical_name);
    for (const std::string prefix : {"unet.", "vae.", "text_encoder.", "text_encoder_2."}) {
        if (starts_with(slot.logical_name, prefix)) {
            add_key(slot.logical_name.substr(prefix.size()));
        }
    }

    const std::vector<std::string> snapshot = prefixes;
    for (const std::string& prefix : snapshot) {
        if (starts_with(prefix, "model.")) insert(prefix.substr(6));
        else insert("model." + prefix);
    }
    return prefixes;
}

[[nodiscard]] std::string tensor_u8_string(const TensorView& view) {
    if (view.dtype != DType::U8 || view.shape.size() != 1 || !view.contiguous()) return {};
    return std::string(reinterpret_cast<const char*>(view.data),
                       static_cast<std::size_t>(view.element_count()));
}

[[nodiscard]] TensorView quantization_scale_for_slot(
    TensorView scale, const ParameterSlot& slot) {
    if (slot.expected_shape.empty()) return scale;
    const std::uint64_t output_rows = slot.expected_shape.front();
    if (scale.element_count() == output_rows) return scale.squeeze_trailing_ones();

    // Original OpenCLIP checkpoints store Q/K/V in one in_proj_weight. The
    // C++ graph exposes zero-copy row slices, so a prequantized checkpoint's
    // per-row scale vector must receive the identical slice.
    if (scale.element_count() == 3 * output_rows && !scale.shape.empty() &&
        scale.shape.front() == 3 * output_rows) {
        std::size_t index = 3;
        if (ends_with(slot.logical_name, "self_attn.q_proj.weight")) index = 0;
        else if (ends_with(slot.logical_name, "self_attn.k_proj.weight")) index = 1;
        else if (ends_with(slot.logical_name, "self_attn.v_proj.weight")) index = 2;
        if (index < 3) {
            return scale.slice(0, index * output_rows, output_rows)
                        .squeeze_trailing_ones();
        }
    }
    return scale;
}

[[nodiscard]] std::optional<QuantizationMetadata> bind_quantization_metadata(
    const WeightStore& store, const ParameterSlot& slot, const TensorView& bound) {
    if (slot.expected_shape.size() != 2 || !ends_with(slot.logical_name, ".weight")) {
        return std::nullopt;
    }

    QuantizationMetadata metadata;
    for (const std::string& prefix : quantization_prefix_candidates(slot, bound)) {
        if (!metadata.weight_scale.has_value()) {
            if (const TensorView* scale = store.find(prefix + "weight_scale")) {
                metadata.weight_scale = quantization_scale_for_slot(*scale, slot);
            }
        }
        if (!metadata.input_scale.has_value()) {
            if (const TensorView* scale = store.find(prefix + "input_scale")) {
                metadata.input_scale = *scale;
            }
        }
        if (const TensorView* config = store.find(prefix + "comfy_quant")) {
            try {
                const std::string text = tensor_u8_string(*config);
                if (!text.empty()) {
                    const json::Value parsed = json::parse(text);
                    if (const json::Value* value = parsed.find("convrot")) {
                        metadata.convrot = value->as_bool();
                    }
                    if (const json::Value* value = parsed.find("per_row")) {
                        metadata.per_row = value->as_bool();
                    }
                    if (const json::Value* value = parsed.find("convrot_groupsize")) {
                        metadata.convrot_group_size = static_cast<std::size_t>(value->as_u64());
                    }
                }
            } catch (const std::exception&) {
                // Ignore malformed optional metadata here. The native INT8 loader
                // performs strict validation when an INT8 precision profile is selected.
            }
            break;
        }
    }

    if (!metadata.weight_scale.has_value() && !metadata.input_scale.has_value() &&
        !metadata.convrot && metadata.convrot_group_size == 0) {
        return std::nullopt;
    }
    if (metadata.weight_scale.has_value() && !metadata.per_row) {
        const TensorView& scale = *metadata.weight_scale;
        metadata.per_row = scale.element_count() == slot.expected_shape[0];
    }
    if (metadata.convrot && metadata.convrot_group_size == 0) {
        metadata.convrot_group_size = 256;
    }
    return metadata;
}

[[nodiscard]] std::optional<TensorView> bind_original_parameter(const WeightStore& store,
                                                                const ParameterSlot& slot) {
    if (const TensorView* direct = store.find(slot.logical_name)) return *direct;

    std::string componentless;
    for (const std::string prefix : {"unet.", "vae.", "text_encoder.", "text_encoder_2."}) {
        if (starts_with(slot.logical_name, prefix)) {
            componentless = slot.logical_name.substr(prefix.size());
            break;
        }
    }
    if (!componentless.empty()) {
        if (const TensorView* direct = store.find(componentless)) return *direct;
    }

    if (starts_with(slot.logical_name, "unet.")) {
        for (const auto& key : unet_original_candidates(slot.logical_name)) {
            if (const TensorView* view = store.find(key)) return apply_transform(*view, DerivedLookup::Transform::None, slot.expected_shape);
        }
    } else if (starts_with(slot.logical_name, "vae.")) {
        for (const auto& key : vae_original_candidates(slot.logical_name)) {
            if (const TensorView* view = store.find(key)) return apply_transform(*view, DerivedLookup::Transform::None, slot.expected_shape);
        }
    } else if (starts_with(slot.logical_name, "text_encoder.")) {
        const std::string key = strip_prefix(slot.logical_name, "text_encoder.");
        for (const auto& prefix : {"conditioner.embedders.0.transformer.", "conditioner.embedders.0.model."}) {
            if (const TensorView* view = store.find(std::string(prefix) + key)) return *view;
        }
    } else if (starts_with(slot.logical_name, "text_encoder_2.")) {
        const std::string key = strip_prefix(slot.logical_name, "text_encoder_2.");
        // Some converted single files already store Hugging Face CLIP keys under a component prefix.
        for (const auto& prefix : {"conditioner.embedders.1.model.", "conditioner.embedders.1.transformer."}) {
            if (const TensorView* view = store.find(std::string(prefix) + key)) return *view;
        }
        for (const auto& lookup : text_encoder_2_original_candidates(slot.logical_name)) {
            if (const TensorView* view = store.find(lookup.key)) {
                return apply_transform(*view, lookup.transform, slot.expected_shape);
            }
        }
    }

    if (!componentless.empty()) {
        return unique_suffix_lookup(store, componentless);
    }
    return std::nullopt;
}

[[nodiscard]] json::Value read_json_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw Error("cannot open JSON file: " + path.string());
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return json::parse(buffer.str());
}

[[nodiscard]] std::vector<std::filesystem::path> component_weight_files(
    const std::filesystem::path& directory,
    bool prefer_fp16) {
    if (!std::filesystem::is_directory(directory)) {
        throw Error("missing SDXL component directory: " + directory.string());
    }

    std::vector<std::filesystem::path> index_files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && ends_with(entry.path().filename().string(), ".safetensors.index.json")) {
            index_files.push_back(entry.path());
        }
    }
    std::sort(index_files.begin(), index_files.end());
    if (!index_files.empty()) {
        std::filesystem::path chosen = index_files.front();
        if (prefer_fp16) {
            for (const auto& path : index_files) {
                if (path.filename().string().find("fp16") != std::string::npos) {
                    chosen = path;
                    break;
                }
            }
        }
        const json::Value index = read_json_file(chosen);
        const json::Value& weight_map = index.at("weight_map");
        std::set<std::filesystem::path> shards;
        for (const auto& [_, file] : weight_map.as_object()) {
            shards.insert(directory / file.as_string());
        }
        return {shards.begin(), shards.end()};
    }

    std::vector<std::filesystem::path> all;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") all.push_back(entry.path());
    }
    std::sort(all.begin(), all.end());
    if (all.empty()) throw Error("no safetensors weights in component directory: " + directory.string());

    auto score = [&](const std::filesystem::path& path) {
        const std::string name = path.filename().string();
        int value = 0;
        if (name.find("fp16") != std::string::npos) value += prefer_fp16 ? 100 : -10;
        if (name == "model.safetensors" || name == "diffusion_pytorch_model.safetensors") value += 50;
        if (name == "model.fp16.safetensors" || name == "diffusion_pytorch_model.fp16.safetensors") value += 60;
        return value;
    };
    const auto best = std::max_element(all.begin(), all.end(), [&](const auto& a, const auto& b) {
        return score(a) < score(b);
    });
    return {*best};
}

struct ComponentStores {
    std::map<std::string, std::unique_ptr<WeightStore>, std::less<>> stores;
    std::size_t file_count = 0;
    std::size_t tensor_count = 0;
};

[[nodiscard]] ComponentStores load_diffusers_stores(const std::filesystem::path& root,
                                                     bool prefer_fp16) {
    ComponentStores result;
    for (const std::string component : {"text_encoder", "text_encoder_2", "unet", "vae"}) {
        auto store = std::make_unique<WeightStore>();
        for (const auto& file : component_weight_files(root / component, prefer_fp16)) {
            store->add_file(file);
        }
        result.file_count += store->files().size();
        result.tensor_count += store->tensor_count();
        result.stores.emplace(component, std::move(store));
    }
    return result;
}

[[nodiscard]] std::string component_for_logical(std::string_view logical) {
    const auto dot = logical.find('.');
    return dot == std::string_view::npos ? std::string(logical) : std::string(logical.substr(0, dot));
}

[[nodiscard]] std::string component_key(std::string_view logical) {
    const auto dot = logical.find('.');
    return dot == std::string_view::npos ? std::string() : std::string(logical.substr(dot + 1));
}

} // namespace

SDXLConfig SDXLConfig::base_1_0() {
    SDXLConfig cfg;
    cfg.clip_l = TextEncoder{
        49408, 77, 768, 3072, 12, 12, 768, false, "quick_gelu"
    };
    cfg.openclip_big_g = TextEncoder{
        49408, 77, 1280, 5120, 32, 20, 1280, true, "gelu"
    };
    cfg.unet_channels = {320, 640, 1280};
    cfg.unet_heads = {5, 10, 20};
    cfg.unet_transformer_depth = {1, 2, 10};
    cfg.unet_cross_attention_dim = 2048;
    cfg.unet_time_embed_dim = 1280;
    cfg.unet_addition_time_embed_dim = 256;
    cfg.unet_addition_input_dim = 2816;
    cfg.unet_layers_per_block = 2;
    cfg.unet_norm_groups = 32;
    cfg.vae_channels = {128, 256, 512, 512};
    cfg.vae_latent_channels = 4;
    cfg.vae_layers_per_block = 2;
    cfg.vae_norm_groups = 32;
    cfg.vae_scaling_factor = 0.13025F;
    return cfg;
}

SDXLModel::SDXLModel(SDXLConfig config) : config_(std::move(config)) {
    add_clip_text_encoder(graph_.root(), "text_encoder", config_.clip_l);
    add_clip_text_encoder(graph_.root(), "text_encoder_2", config_.openclip_big_g);
    add_unet(graph_.root(), config_);
    add_vae(graph_.root(), config_);
    graph_.rebuild_index();
}

SDXLWeightLoader::SDXLWeightLoader(LoadOptions options) : options_(options) {}

LoadResult SDXLWeightLoader::load(SDXLModel& model, const std::filesystem::path& path) {
    LoadResult result;
    for (auto& [_, slot] : model.graph().parameter_index()) {
        slot->tensor.reset();
        slot->quantization.reset();
    }

    if (std::filesystem::is_directory(path)) {
        result.layout = CheckpointLayout::DiffusersDirectory;
        ComponentStores components = load_diffusers_stores(path, options_.prefer_fp16_variant);
        result.files_mapped = components.file_count;
        result.tensors_discovered = components.tensor_count;

        for (auto& [logical, slot] : model.graph().parameter_index()) {
            const std::string component = component_for_logical(logical);
            const auto store_it = components.stores.find(component);
            if (store_it == components.stores.end()) continue;
            const std::string key = component_key(logical);
            if (const TensorView* view = store_it->second->find(key)) {
                slot->tensor = *view;
                slot->quantization = bind_quantization_metadata(*store_it->second, *slot, *view);
                ++result.parameters_bound;
            }
        }
    } else if (std::filesystem::is_regular_file(path) && path.extension() == ".safetensors") {
        result.layout = CheckpointLayout::OriginalSingleFile;
        WeightStore store;
        store.add_file(path);
        result.files_mapped = 1;
        result.tensors_discovered = store.tensor_count();

        for (auto& [_, slot] : model.graph().parameter_index()) {
            if (auto view = bind_original_parameter(store, *slot)) {
                if (view->shape != slot->expected_shape) {
                    TensorView squeezed = view->squeeze_trailing_ones();
                    if (squeezed.shape == slot->expected_shape) *view = std::move(squeezed);
                }
                slot->quantization = bind_quantization_metadata(store, *slot, *view);
                slot->tensor = std::move(*view);
                ++result.parameters_bound;
            }
        }
    } else {
        throw Error("SDXL weight path must be a Diffusers directory or a .safetensors file: " + path.string());
    }

    result.validation = model.graph().validate(true);
    if (result.layout == CheckpointLayout::OriginalSingleFile) {
        result.notes.push_back("Original SDXL QKV tensors are exposed as zero-copy row slices.");
        result.notes.push_back("OpenCLIP text_projection is exposed as a strided transpose view.");
        result.notes.push_back("Original VAE 1x1 attention convolutions are exposed as squeezed 2D views.");
    }

    if (options_.strict && !result.validation.ok()) {
        std::ostringstream stream;
        stream << "SDXL checkpoint validation failed: " << result.validation.missing
               << " missing, " << result.validation.shape_mismatches << " shape mismatches";
        if (!result.validation.issues.empty()) {
            stream << ". First issue: " << result.validation.issues.front().logical_name
                   << " - " << result.validation.issues.front().detail;
        }
        throw Error(stream.str());
    }
    return result;
}

} // namespace sdxl
