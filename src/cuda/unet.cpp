#include "sdxl/cuda/unet.hpp"
#include "sdxl/cuda/profiler.hpp"

#include <string>
#include <utility>
#include <vector>

namespace sdxl::cuda {

UNet::UNet(const Runtime& runtime,
           const SDXLModel& model,
           const WeightStore& weights,
           UNetOptions options)
    : runtime_(&runtime),
      model_(&model),
      weights_(&weights),
      ops_(runtime),
      options_(options) {}

Tensor UNet::build_time_embedding(const Tensor& latents,
                                  float timestep,
                                  const Tensor& pooled_text_embeds,
                                  const Tensor& time_ids) const {
    auto profile = profile_scope("unet/time_embedding");
    const std::size_t batch = latents.size(0);
    Tensor time_projection = ops_.timestep_embedding_scalar(
        timestep, batch, static_cast<std::size_t>(model_->config().unet_channels.front()));
    Tensor time_embedding = ops_.linear_activation(
        time_projection,
        weights_->get("unet.time_embedding.linear_1.weight"),
        weights_->get("unet.time_embedding.linear_1.bias"),
        LinearActivation::SiLU);
    time_embedding = ops_.linear(
        time_embedding,
        weights_->get("unet.time_embedding.linear_2.weight"),
        &weights_->get("unet.time_embedding.linear_2.bias"));

    Tensor additional_time = ops_.timestep_embedding_values(
        time_ids, static_cast<std::size_t>(model_->config().unet_addition_time_embed_dim));
    Tensor additional_input = ops_.concat_last_dim(pooled_text_embeds, additional_time);
    Tensor additional_embedding = ops_.linear_activation(
        additional_input,
        weights_->get("unet.add_embedding.linear_1.weight"),
        weights_->get("unet.add_embedding.linear_1.bias"),
        LinearActivation::SiLU);
    additional_embedding = ops_.linear(
        additional_embedding,
        weights_->get("unet.add_embedding.linear_2.weight"),
        &weights_->get("unet.add_embedding.linear_2.bias"));
    return ops_.add_silu(time_embedding, additional_embedding);
}

Tensor UNet::resnet_block(std::string_view prefix,
                          const Tensor& input,
                          const Tensor& activated_temb) const {
    auto profile = profile_scope(std::string("unet/resnet/") + std::string(prefix));
    const std::string base(prefix);
    Tensor hidden = ops_.group_norm_silu_nchw(
        input,
        weights_->get(base + ".norm1.weight"),
        weights_->get(base + ".norm1.bias"),
        static_cast<std::size_t>(model_->config().unet_norm_groups));
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get(base + ".conv1.weight"),
        &weights_->get(base + ".conv1.bias"),
        1, 1, 1, 1);

    Tensor projected_temb = ops_.linear(
        activated_temb,
        weights_->get(base + ".time_emb_proj.weight"),
        &weights_->get(base + ".time_emb_proj.bias"));
    hidden = ops_.add_spatial_bias(hidden, projected_temb);

    hidden = ops_.group_norm_silu_nchw(
        hidden,
        weights_->get(base + ".norm2.weight"),
        weights_->get(base + ".norm2.bias"),
        static_cast<std::size_t>(model_->config().unet_norm_groups));
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get(base + ".conv2.weight"),
        &weights_->get(base + ".conv2.bias"),
        1, 1, 1, 1);

    Tensor residual = input;
    if (weights_->contains(base + ".conv_shortcut.weight")) {
        residual = ops_.convolution_nchw(
            input,
            weights_->get(base + ".conv_shortcut.weight"),
            &weights_->get(base + ".conv_shortcut.bias"),
            1, 1, 0, 0);
    }
    ops_.add_in_place(hidden, residual);
    return hidden;
}

Tensor UNet::transformer_block(std::string_view prefix,
                               const Tensor& input,
                               const Tensor& encoder_hidden_states,
                               std::size_t heads) const {
    auto profile = profile_scope(std::string("unet/transformer_block/") + std::string(prefix));
    const std::string base(prefix);
    Tensor hidden = input;

    Tensor normalized = ops_.layer_norm(
        hidden,
        weights_->get(base + ".norm1.weight"),
        weights_->get(base + ".norm1.bias"));
    Tensor query = ops_.linear(normalized, weights_->get(base + ".attn1.to_q.weight"));
    Tensor key = ops_.linear(normalized, weights_->get(base + ".attn1.to_k.weight"));
    Tensor value = ops_.linear(normalized, weights_->get(base + ".attn1.to_v.weight"));
    Tensor attention = ops_.attention(query, key, value, heads, false, nullptr);
    attention = ops_.linear(
        attention,
        weights_->get(base + ".attn1.to_out.0.weight"),
        &weights_->get(base + ".attn1.to_out.0.bias"));
    ops_.add_in_place(hidden, attention);

    normalized = ops_.layer_norm(
        hidden,
        weights_->get(base + ".norm2.weight"),
        weights_->get(base + ".norm2.bias"));
    query = ops_.linear(normalized, weights_->get(base + ".attn2.to_q.weight"));
    key = ops_.linear(encoder_hidden_states, weights_->get(base + ".attn2.to_k.weight"));
    value = ops_.linear(encoder_hidden_states, weights_->get(base + ".attn2.to_v.weight"));
    attention = ops_.attention(query, key, value, heads, false, nullptr);
    attention = ops_.linear(
        attention,
        weights_->get(base + ".attn2.to_out.0.weight"),
        &weights_->get(base + ".attn2.to_out.0.bias"));
    ops_.add_in_place(hidden, attention);

    normalized = ops_.layer_norm(
        hidden,
        weights_->get(base + ".norm3.weight"),
        weights_->get(base + ".norm3.bias"));
    Tensor feed_forward = ops_.linear_activation(
        normalized,
        weights_->get(base + ".ff.net.0.proj.weight"),
        weights_->get(base + ".ff.net.0.proj.bias"),
        LinearActivation::GEGLU);
    feed_forward = ops_.linear(
        feed_forward,
        weights_->get(base + ".ff.net.2.weight"),
        &weights_->get(base + ".ff.net.2.bias"));
    ops_.add_in_place(hidden, feed_forward);
    return hidden;
}

Tensor UNet::transformer2d(std::string_view prefix,
                           const Tensor& input,
                           const Tensor& encoder_hidden_states,
                           std::size_t depth,
                           std::size_t heads) const {
    auto profile = profile_scope(std::string("unet/attention_block/") + std::string(prefix));
    const std::string base(prefix);
    Tensor hidden = ops_.group_norm_nchw(
        input,
        weights_->get(base + ".norm.weight"),
        weights_->get(base + ".norm.bias"),
        static_cast<std::size_t>(model_->config().unet_norm_groups),
        1.0e-6F);
    hidden = ops_.flatten_spatial(hidden);
    hidden = ops_.linear(
        hidden,
        weights_->get(base + ".proj_in.weight"),
        &weights_->get(base + ".proj_in.bias"));
    for (std::size_t layer = 0; layer < depth; ++layer) {
        hidden = transformer_block(
            base + ".transformer_blocks." + std::to_string(layer),
            hidden, encoder_hidden_states, heads);
    }
    hidden = ops_.linear(
        hidden,
        weights_->get(base + ".proj_out.weight"),
        &weights_->get(base + ".proj_out.bias"));
    hidden = ops_.unflatten_spatial(hidden, input.size(2), input.size(3));
    ops_.add_in_place(hidden, input);
    return hidden;
}

Tensor UNet::forward(const Tensor& latent_sample,
                     float timestep,
                     const Tensor& encoder_hidden_states,
                     const Tensor& pooled_text_embeds,
                     const Tensor& time_ids) const {
    auto full_profile = profile_scope("unet/forward");
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA UNet is not initialized");
    }
    if (latent_sample.rank() != 4 || latent_sample.size(1) != 4 ||
        encoder_hidden_states.rank() != 3 ||
        encoder_hidden_states.size(2) != model_->config().unet_cross_attention_dim ||
        pooled_text_embeds.rank() != 2 || time_ids.rank() != 2 || time_ids.size(1) != 6) {
        throw CudaError("CUDA UNet input shape mismatch");
    }
    const std::size_t batch = latent_sample.size(0);
    if (encoder_hidden_states.size(0) != batch || pooled_text_embeds.size(0) != batch ||
        time_ids.size(0) != batch) {
        throw CudaError("CUDA UNet conditioning batch mismatch");
    }

    Tensor activated_temb = build_time_embedding(
        latent_sample, timestep, pooled_text_embeds, time_ids);
    Tensor hidden = ops_.convolution_nchw(
        latent_sample,
        weights_->get("unet.conv_in.weight"),
        &weights_->get("unet.conv_in.bias"),
        1, 1, 1, 1);

    std::vector<Tensor> residuals;
    residuals.reserve(16);
    residuals.push_back(hidden);

    const std::size_t block_count = model_->config().unet_channels.size();
    const std::size_t layers_per_block =
        static_cast<std::size_t>(model_->config().unet_layers_per_block);

    for (std::size_t block = 0; block < block_count; ++block) {
        auto block_profile = profile_scope("unet/down_block/" + std::to_string(block));
        const bool with_attention = block > 0;
        const std::size_t depth =
            static_cast<std::size_t>(model_->config().unet_transformer_depth[block]);
        const std::size_t heads =
            static_cast<std::size_t>(model_->config().unet_heads[block]);
        for (std::size_t layer = 0; layer < layers_per_block; ++layer) {
            const std::string prefix = "unet.down_blocks." + std::to_string(block);
            hidden = resnet_block(
                prefix + ".resnets." + std::to_string(layer), hidden, activated_temb);
            if (with_attention) {
                hidden = transformer2d(
                    prefix + ".attentions." + std::to_string(layer),
                    hidden, encoder_hidden_states, depth, heads);
            }
            residuals.push_back(hidden);
        }
        if (block + 1 < block_count) {
            const std::string prefix = "unet.down_blocks." + std::to_string(block) +
                                       ".downsamplers.0.conv";
            hidden = ops_.convolution_nchw(
                hidden, weights_->get(prefix + ".weight"),
                &weights_->get(prefix + ".bias"), 2, 2, 1, 1);
            residuals.push_back(hidden);
        }
    }

    {
    auto mid_profile = profile_scope("unet/mid_block");
    hidden = resnet_block("unet.mid_block.resnets.0", hidden, activated_temb);
    hidden = transformer2d(
        "unet.mid_block.attentions.0", hidden, encoder_hidden_states,
        static_cast<std::size_t>(model_->config().unet_transformer_depth.back()),
        static_cast<std::size_t>(model_->config().unet_heads.back()));
    hidden = resnet_block("unet.mid_block.resnets.1", hidden, activated_temb);
    }

    for (std::size_t block = 0; block < block_count; ++block) {
        auto block_profile = profile_scope("unet/up_block/" + std::to_string(block));
        const std::size_t down_index = block_count - 1 - block;
        const std::size_t depth =
            static_cast<std::size_t>(model_->config().unet_transformer_depth[down_index]);
        const std::size_t heads =
            static_cast<std::size_t>(model_->config().unet_heads[down_index]);
        for (std::size_t layer = 0; layer < layers_per_block + 1; ++layer) {
            if (residuals.empty()) throw CudaError("CUDA UNet residual stack underflow");
            Tensor skip = residuals.back();
            residuals.pop_back();
            hidden = ops_.concat_channels(hidden, skip);
            const std::string prefix = "unet.up_blocks." + std::to_string(block);
            hidden = resnet_block(
                prefix + ".resnets." + std::to_string(layer), hidden, activated_temb);
            if (down_index > 0) {
                hidden = transformer2d(
                    prefix + ".attentions." + std::to_string(layer),
                    hidden, encoder_hidden_states, depth, heads);
            }
        }
        if (block + 1 < block_count) {
            if (residuals.empty()) throw CudaError("CUDA UNet cannot resolve upsample target");
            hidden = ops_.nearest_upsample(
                hidden, residuals.back().size(2), residuals.back().size(3));
            const std::string prefix = "unet.up_blocks." + std::to_string(block) +
                                       ".upsamplers.0.conv";
            hidden = ops_.convolution_nchw(
                hidden, weights_->get(prefix + ".weight"),
                &weights_->get(prefix + ".bias"), 1, 1, 1, 1);
        }
    }
    if (!residuals.empty()) throw CudaError("CUDA UNet left unused residual tensors");

    {
    auto output_profile = profile_scope("unet/output_head");
    hidden = ops_.group_norm_silu_nchw(
        hidden,
        weights_->get("unet.conv_norm_out.weight"),
        weights_->get("unet.conv_norm_out.bias"),
        static_cast<std::size_t>(model_->config().unet_norm_groups));
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get("unet.conv_out.weight"),
        &weights_->get("unet.conv_out.bias"),
        1, 1, 1, 1);
    }
    if (hidden.shape() != latent_sample.shape()) {
        throw CudaError("CUDA UNet output shape differs from latent input");
    }
    if (options_.check_finite_output) ops_.check_finite(hidden, "CUDA UNet output");
    return hidden;
}

} // namespace sdxl::cuda
