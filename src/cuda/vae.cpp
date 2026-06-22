#include "sdxl/cuda/vae.hpp"
#include "sdxl/cuda/profiler.hpp"

#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace sdxl::cuda {

VAE::VAE(const Runtime& runtime,
         const SDXLModel& model,
         const WeightStore& weights,
         VAEOptions options)
    : runtime_(&runtime),
      model_(&model),
      weights_(&weights),
      ops_(runtime),
      options_(options) {
    if (!(options_.scaling_factor > 0.0F) || !std::isfinite(options_.scaling_factor)) {
        throw CudaError("VAE scaling factor must be finite and positive");
    }
    if (options_.norm_groups == 0) throw CudaError("VAE norm group count must be positive");
}

Tensor VAE::resnet_block(std::string_view prefix, const Tensor& input) const {
    auto profile = profile_scope(std::string("vae/resnet/") + std::string(prefix));
    const std::string base(prefix);
    Tensor hidden = ops_.group_norm_silu_nchw(
        input,
        weights_->get(base + ".norm1.weight"),
        weights_->get(base + ".norm1.bias"),
        options_.norm_groups,
        1.0e-6F);
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get(base + ".conv1.weight"),
        &weights_->get(base + ".conv1.bias"),
        1, 1, 1, 1);

    hidden = ops_.group_norm_silu_nchw(
        hidden,
        weights_->get(base + ".norm2.weight"),
        weights_->get(base + ".norm2.bias"),
        options_.norm_groups,
        1.0e-6F);
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

Tensor VAE::attention_block(std::string_view prefix, const Tensor& input) const {
    auto profile = profile_scope(std::string("vae/attention/") + std::string(prefix));
    const std::string base(prefix);
    Tensor hidden = ops_.group_norm_nchw(
        input,
        weights_->get(base + ".group_norm.weight"),
        weights_->get(base + ".group_norm.bias"),
        options_.norm_groups,
        1.0e-6F);
    hidden = ops_.flatten_spatial(hidden);

    Tensor query = ops_.linear(
        hidden,
        weights_->get(base + ".to_q.weight"),
        &weights_->get(base + ".to_q.bias"));
    Tensor key = ops_.linear(
        hidden,
        weights_->get(base + ".to_k.weight"),
        &weights_->get(base + ".to_k.bias"));
    Tensor value = ops_.linear(
        hidden,
        weights_->get(base + ".to_v.weight"),
        &weights_->get(base + ".to_v.bias"));

    // The SDXL VAE mid-block uses a single attention head whose width equals
    // the channel count (512 for the base model). Ops::attention dispatches
    // this to the generic memory-efficient online-softmax CUDA kernel.
    hidden = ops_.attention(query, key, value, 1, false, nullptr);
    hidden = ops_.linear(
        hidden,
        weights_->get(base + ".to_out.0.weight"),
        &weights_->get(base + ".to_out.0.bias"));
    hidden = ops_.unflatten_spatial(hidden, input.size(2), input.size(3));
    ops_.add_in_place(hidden, input);
    return hidden;
}

Tensor VAE::decode(const Tensor& latents) const {
    auto decode_profile = profile_scope("vae/decode_full");
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA VAE is not initialized");
    }
    if ((latents.type() != ScalarType::Float16 && latents.type() != ScalarType::Float32) ||
        latents.rank() != 4 ||
        latents.size(1) != static_cast<std::size_t>(model_->config().vae_latent_channels)) {
        throw CudaError("CUDA VAE expects float [B,4,H,W] SDXL latents");
    }
    if (!weights_->contains("vae.post_quant_conv.weight") ||
        !weights_->contains("vae.decoder.conv_out.weight")) {
        throw CudaError("CUDA VAE weights are not resident; load prefix vae. first");
    }

    // SDXL's base VAE is configured with force_upcast=true. Keep the denoising
    // latents in FP16, then upcast once before every VAE convolution/attention op.
    Tensor hidden = ops_.cast_scale(
        latents, ScalarType::Float32, TensorRole::VAE,
        1.0F / options_.scaling_factor);
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get("vae.post_quant_conv.weight"),
        &weights_->get("vae.post_quant_conv.bias"),
        1, 1, 0, 0);
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get("vae.decoder.conv_in.weight"),
        &weights_->get("vae.decoder.conv_in.bias"),
        1, 1, 1, 1);

    hidden = resnet_block("vae.decoder.mid_block.resnets.0", hidden);
    hidden = attention_block("vae.decoder.mid_block.attentions.0", hidden);
    hidden = resnet_block("vae.decoder.mid_block.resnets.1", hidden);

    const std::size_t block_count = model_->config().vae_channels.size();
    const std::size_t layers = static_cast<std::size_t>(model_->config().vae_layers_per_block) + 1;
    for (std::size_t block = 0; block < block_count; ++block) {
        auto block_profile = profile_scope("vae/up_block/" + std::to_string(block));
        const std::string block_prefix = "vae.decoder.up_blocks." + std::to_string(block);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            hidden = resnet_block(
                block_prefix + ".resnets." + std::to_string(layer), hidden);
        }
        if (block + 1 < block_count) {
            hidden = ops_.nearest_upsample(
                hidden, hidden.size(2) * 2, hidden.size(3) * 2);
            hidden = ops_.convolution_nchw(
                hidden,
                weights_->get(block_prefix + ".upsamplers.0.conv.weight"),
                &weights_->get(block_prefix + ".upsamplers.0.conv.bias"),
                1, 1, 1, 1);
        }
    }

    hidden = ops_.group_norm_silu_nchw(
        hidden,
        weights_->get("vae.decoder.conv_norm_out.weight"),
        weights_->get("vae.decoder.conv_norm_out.bias"),
        options_.norm_groups,
        1.0e-6F);
    hidden = ops_.convolution_nchw(
        hidden,
        weights_->get("vae.decoder.conv_out.weight"),
        &weights_->get("vae.decoder.conv_out.bias"),
        1, 1, 1, 1);

    if (hidden.rank() != 4 || hidden.size(1) != 3 ||
        hidden.size(2) != latents.size(2) * 8 ||
        hidden.size(3) != latents.size(3) * 8) {
        throw CudaError("CUDA VAE produced an unexpected RGB tensor shape");
    }
    if (options_.check_finite_output) ops_.check_finite(hidden, "CUDA VAE decoder output");
    return hidden;
}

} // namespace sdxl::cuda
