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

        sdxl::EulerDiscreteScheduler euler;
        euler.set_timesteps(4);
        if (euler.timesteps().size() != 4 || euler.sigmas().size() != 5) return 3;
        if (!near(euler.initial_noise_sigma(), 14.6146F, 2.0e-3F)) return 4;
        for (std::size_t index = 1; index < euler.sigmas().size(); ++index) {
            if (euler.sigmas()[index] > euler.sigmas()[index - 1]) return 5;
        }
        sdxl::FloatTensor scheduler_sample{{1, 1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        sdxl::FloatTensor zero_prediction{{1, 1, 2, 2}, {0.0F, 0.0F, 0.0F, 0.0F}};
        const sdxl::FloatTensor unchanged = euler.step(zero_prediction, 0, scheduler_sample);
        for (std::size_t index = 0; index < unchanged.values.size(); ++index) {
            if (!near(unchanged.values[index], scheduler_sample.values[index])) return 6;
        }

        const sdxl::FloatTensor random_a = sdxl::random_normal_tensor({1, 4, 2, 2}, 1234);
        const sdxl::FloatTensor random_b = sdxl::random_normal_tensor({1, 4, 2, 2}, 1234);
        if (random_a.values != random_b.values) return 7;

        sdxl::DDIMScheduler ddim;
        ddim.set_timesteps(4);
        const sdxl::FloatTensor ddim_output = ddim.step(
            zero_prediction, 0, scheduler_sample, 0.0F, nullptr);
        if (ddim_output.shape != scheduler_sample.shape) return 8;
        for (const float value : ddim_output.values) {
            if (!std::isfinite(value)) return 9;
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
        if (output.shape != latent.shape) return 10;
        for (const float value : output.values) {
            if (!near(value, 0.25F, 1.0e-4F)) return 11;
        }

        sdxl::SDXLClassifierFreeConditioning conditioning;
        conditioning.positive.prompt_embeds = encoder;
        conditioning.positive.pooled_prompt_embeds = pooled;
        conditioning.negative.prompt_embeds = encoder;
        conditioning.negative.pooled_prompt_embeds = pooled;
        sdxl::SDXLDenoiseOptions options;
        options.width = 32;
        options.height = 32;
        options.inference_steps = 1;
        options.guidance_scale = 2.0F;
        options.seed = 9;
        sdxl::SDXLDenoiser denoiser(model);
        const sdxl::SDXLDenoiseResult result = denoiser.denoise(conditioning, options);
        if (result.latents.shape != latent.shape || result.timesteps.size() != 1) return 12;
        for (const float value : result.latents.values) {
            if (!std::isfinite(value)) return 13;
        }

        std::cout << "UNet and scheduler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 100;
    }
}
