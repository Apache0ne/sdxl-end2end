#include "sdxl/cuda/ops.hpp"
#include "sdxl/cuda/denoise_graph.hpp"
#include "sdxl/cuda/runtime.hpp"
#include "sdxl/cuda/tensor.hpp"
#include "sdxl/cuda/image.hpp"
#include "sdxl/cuda/vae.hpp"
#include "sdxl/cuda/weights.hpp"
#include "sdxl/sdxl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(float actual, float expected, float tolerance, const char* label) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + " mismatch: " +
                                 std::to_string(actual) + " vs " + std::to_string(expected));
    }
}

sdxl::cuda::Tensor upload(sdxl::cuda::Runtime& runtime,
                          std::vector<std::size_t> shape,
                          std::vector<float> values) {
    return sdxl::cuda::Tensor::from_host_f32(
        runtime, sdxl::FloatTensor{std::move(shape), std::move(values)});
}

class OwnedWeights final {
public:
    void bind_vae(sdxl::SDXLModel& model) { bind_prefix(model, "vae."); }
    void bind_unet(sdxl::SDXLModel& model) { bind_prefix(model, "unet."); }

private:
    void bind_prefix(sdxl::SDXLModel& model, const std::string& prefix) {
        for (auto& [name, slot] : model.graph().parameter_index()) {
            if (name.rfind(prefix, 0) != 0) continue;
            std::size_t count = 1;
            for (const std::uint64_t dimension : slot->expected_shape) {
                count *= static_cast<std::size_t>(dimension);
            }
            auto storage = std::make_unique<std::vector<float>>(count, 0.0F);
            if (name.ends_with(".weight") &&
                (name.find(".norm1.weight") != std::string::npos ||
                 name.find(".norm2.weight") != std::string::npos ||
                 name.find(".group_norm.weight") != std::string::npos ||
                 name.find("conv_norm_out.weight") != std::string::npos)) {
                std::fill(storage->begin(), storage->end(), 1.0F);
            }
            if (name == "unet.time_embedding.linear_1.weight") {
                for (std::size_t index = 0; index < storage->size(); ++index) {
                    (*storage)[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.03125F;
                }
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
    config.unet_addition_input_dim = 32;
    config.unet_layers_per_block = 2;
    config.unet_norm_groups = 32;
    config.vae_channels = {32, 32, 32, 32};
    config.vae_latent_channels = 4;
    config.vae_layers_per_block = 2;
    config.vae_norm_groups = 32;
    config.vae_scaling_factor = 0.13025F;
    return config;
}

} // namespace

int main() {
    try {
        sdxl::cuda::Runtime runtime;
        sdxl::cuda::Ops ops(runtime);

        bool rejected_non_vae_fp32 = false;
        try {
            auto forbidden = sdxl::cuda::Tensor::allocate(
                runtime, {1}, sdxl::cuda::ScalarType::Float32,
                sdxl::cuda::TensorRole::Model);
            (void)forbidden;
        } catch (const sdxl::cuda::CudaError&) {
            rejected_non_vae_fp32 = true;
        }
        if (!rejected_non_vae_fp32) {
            throw std::runtime_error("strict non-VAE FP32 allocation guard did not fire");
        }

        // Basic custom elementwise kernel.
        auto first_device = upload(runtime, {1, 4}, {1.0F, 2.0F, 3.0F, 4.0F});
        auto second_device = upload(runtime, {1, 4}, {4.0F, 3.0F, 2.0F, 1.0F});
        auto sum = ops.add(first_device, second_device).to_host_f32(runtime);
        for (float value : sum.values) require_close(value, 5.0F, 0.01F, "add");

        // cuBLASLt row-major FP16 GEMM plus the separate CUDA bias kernel.
        auto linear_input = upload(runtime, {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
        auto linear_weight = upload(runtime, {3, 2}, {
            1.0F, 0.0F,
            0.0F, 1.0F,
            1.0F, 1.0F});
        auto linear_bias = upload(runtime, {3}, {0.5F, 0.5F, 0.5F});
        auto linear_device = ops.linear(linear_input, linear_weight, &linear_bias);
        auto linear = linear_device.to_host_f32(runtime);
        const float linear_expected[] = {1.5F, 2.5F, 3.5F, 3.5F, 4.5F, 7.5F};
        for (std::size_t index = 0; index < linear.values.size(); ++index) {
            require_close(linear.values[index], linear_expected[index], 0.03F, "linear");
        }

        // Fused GEMM bias+activation removes one launch from CLIP/UNet MLPs.
        const auto separate_silu = ops.silu(linear_device).to_host_f32(runtime);
        const auto fused_silu = ops.linear_activation(
            linear_input, linear_weight, linear_bias,
            sdxl::cuda::LinearActivation::SiLU).to_host_f32(runtime);
        for (std::size_t index = 0; index < fused_silu.values.size(); ++index) {
            require_close(fused_silu.values[index], separate_silu.values[index],
                          0.03F, "fused linear SiLU");
        }
        auto geglu_weight = upload(runtime, {4, 2}, {
            1.0F, 0.0F, 0.0F, 1.0F,
            0.5F, 0.0F, 0.0F, 0.5F});
        auto geglu_bias = upload(runtime, {4}, {0.1F, 0.2F, 0.3F, 0.4F});
        const auto separate_geglu = ops.geglu(
            ops.linear(linear_input, geglu_weight, &geglu_bias)).to_host_f32(runtime);
        const auto fused_geglu = ops.linear_activation(
            linear_input, geglu_weight, geglu_bias,
            sdxl::cuda::LinearActivation::GEGLU).to_host_f32(runtime);
        for (std::size_t index = 0; index < fused_geglu.values.size(); ++index) {
            require_close(fused_geglu.values[index], separate_geglu.values[index],
                          0.03F, "fused linear GEGLU");
        }

        // FP8 E4M3 UNet linear. On Ada/Hopper this dispatches native FP8 cuBLASLt;
        // on Ampere it keeps FP8 weights resident and decodes tiles into FP16 WMMA.
        auto linear_weight_fp8 = ops.quantize_e4m3(linear_weight);
        if (linear_weight_fp8.type() != sdxl::cuda::ScalarType::Float8E4M3 ||
            !linear_weight_fp8.has_dequant_scale()) {
            throw std::runtime_error("FP8 quantization did not attach a dequantization scale");
        }
        auto linear_fp8 = ops.linear(
            linear_input, linear_weight_fp8, &linear_bias).to_host_f32(runtime);
        for (std::size_t index = 0; index < linear_fp8.values.size(); ++index) {
            require_close(linear_fp8.values[index], linear_expected[index], 0.20F, "linear FP8");
        }

        auto linear_weight_e5m2 = ops.quantize_e5m2(linear_weight);
        if (linear_weight_e5m2.type() != sdxl::cuda::ScalarType::Float8E5M2 ||
            !linear_weight_e5m2.has_dequant_scale()) {
            throw std::runtime_error("E5M2 quantization did not attach a dequantization scale");
        }
        auto linear_e5m2 = ops.linear(
            linear_input, linear_weight_e5m2, &linear_bias).to_host_f32(runtime);
        for (std::size_t index = 0; index < linear_e5m2.values.size(); ++index) {
            require_close(linear_e5m2.values[index], linear_expected[index], 0.35F, "linear E5M2");
        }

        // 16x16 shape selects the native cuBLASLt FP8 path on SM 8.9+ and the
        // fused FP8-weight WMMA path on Ampere.
        std::vector<float> aligned_input_values(16 * 16);
        std::vector<float> aligned_weight_values(16 * 16, 0.0F);
        for (std::size_t row = 0; row < 16; ++row) {
            for (std::size_t column = 0; column < 16; ++column) {
                aligned_input_values[row * 16 + column] =
                    static_cast<float>(static_cast<int>((row + column) % 7) - 3) * 0.125F;
            }
            aligned_weight_values[row * 16 + row] = 1.0F;
        }
        auto aligned_input = upload(runtime, {16, 16}, aligned_input_values);
        auto aligned_weight = upload(runtime, {16, 16}, aligned_weight_values);
        auto aligned_weight_fp8 = ops.quantize_e4m3(aligned_weight);
        auto aligned_output = ops.linear(
            aligned_input, aligned_weight_fp8, nullptr).to_host_f32(runtime);
        for (std::size_t index = 0; index < aligned_output.values.size(); ++index) {
            require_close(aligned_output.values[index], aligned_input_values[index],
                          0.08F, "aligned FP8 linear");
        }

        // FP32/TF32 cuBLASLt path used by the force-upcast VAE.
        auto linear_input_f32 = ops.cast(linear_input, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto linear_weight_f32 = ops.cast(linear_weight, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto linear_bias_f32 = ops.cast(linear_bias, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto linear_f32 = ops.linear(
            linear_input_f32, linear_weight_f32, &linear_bias_f32).to_host_f32(runtime);
        for (std::size_t index = 0; index < linear_f32.values.size(); ++index) {
            require_close(linear_f32.values[index], linear_expected[index], 0.001F, "linear FP32");
        }

        // cuDNN NCHW convolution.
        auto conv_input = upload(runtime, {1, 1, 3, 3}, {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
            7.0F, 8.0F, 9.0F});
        auto conv_weight = upload(runtime, {1, 1, 3, 3}, std::vector<float>(9, 1.0F));
        auto conv_bias = upload(runtime, {1}, {0.0F});
        auto convolution = ops.convolution_nchw(
            conv_input, conv_weight, &conv_bias, 1, 1, 1, 1).to_host_f32(runtime);
        require_close(convolution.values[0], 12.0F, 0.05F, "convolution corner");
        require_close(convolution.values[4], 45.0F, 0.05F, "convolution center");

        // FP32 cuDNN path used by the VAE decoder.
        auto conv_input_f32 = ops.cast(conv_input, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto conv_weight_f32 = ops.cast(conv_weight, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto conv_bias_f32 = ops.cast(conv_bias, sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto convolution_f32 = ops.convolution_nchw(
            conv_input_f32, conv_weight_f32, &conv_bias_f32, 1, 1, 1, 1).to_host_f32(runtime);
        require_close(convolution_f32.values[0], 12.0F, 0.001F, "convolution FP32 corner");
        require_close(convolution_f32.values[4], 45.0F, 0.001F, "convolution FP32 center");

        // FP32-reduction group normalization kernel.
        auto norm_input = upload(runtime, {1, 2, 1, 2}, {1.0F, 3.0F, 5.0F, 7.0F});
        auto norm_weight = upload(runtime, {2}, {1.0F, 1.0F});
        auto norm_bias = upload(runtime, {2}, {0.0F, 0.0F});
        auto normalized = ops.group_norm_nchw(
            norm_input, norm_weight, norm_bias, 1, 1.0e-5F).to_host_f32(runtime);
        require_close(normalized.values[0], -1.34164F, 0.02F, "group norm");
        require_close(normalized.values[3], 1.34164F, 0.02F, "group norm");

        // Stable two-pass LayerNorm regression. The large common offset makes
        // E[x^2] - E[x]^2 cancellation severe while centered variance remains stable.
        std::vector<float> offset_values(320);
        for (std::size_t index = 0; index < offset_values.size(); ++index) {
            offset_values[index] = 1000.0F +
                (static_cast<float>(static_cast<int>(index % 8) - 4) * 0.5F);
        }
        auto offset_input = upload(runtime, {1, 320}, offset_values);
        auto offset_weight = upload(runtime, {320}, std::vector<float>(320, 1.0F));
        auto offset_bias = upload(runtime, {320}, std::vector<float>(320, 0.0F));
        const auto offset_norm = ops.layer_norm(
            offset_input, offset_weight, offset_bias, 1.0e-5F).to_host_f32(runtime);
        float offset_mean = 0.0F;
        for (const float value : offset_norm.values) {
            if (!std::isfinite(value)) throw std::runtime_error("stable LayerNorm produced a non-finite value");
            offset_mean += value;
        }
        offset_mean /= static_cast<float>(offset_norm.values.size());
        float offset_variance = 0.0F;
        for (const float value : offset_norm.values) {
            const float centered = value - offset_mean;
            offset_variance += centered * centered;
        }
        offset_variance /= static_cast<float>(offset_norm.values.size());
        require_close(offset_mean, 0.0F, 0.02F, "stable LayerNorm mean");
        require_close(offset_variance, 1.0F, 0.05F, "stable LayerNorm variance");

        // Fused GroupNorm+SiLU must match the unfused reference sequence.
        const auto norm_then_silu = ops.silu(ops.group_norm_nchw(
            norm_input, norm_weight, norm_bias, 1, 1.0e-5F)).to_host_f32(runtime);
        const auto fused_norm_silu = ops.group_norm_silu_nchw(
            norm_input, norm_weight, norm_bias, 1, 1.0e-5F).to_host_f32(runtime);
        for (std::size_t index = 0; index < fused_norm_silu.values.size(); ++index) {
            require_close(fused_norm_silu.values[index], norm_then_silu.values[index],
                          0.02F, "fused GroupNorm+SiLU");
        }

        // Online-softmax attention specialized for SDXL's head dimension 64.
        sdxl::FloatTensor q{{1, 2, 64}, std::vector<float>(128, 0.0F)};
        sdxl::FloatTensor k = q;
        sdxl::FloatTensor v{{1, 2, 64}, std::vector<float>(128, 0.0F)};
        for (std::size_t i = 0; i < 64; ++i) {
            v.values[i] = 2.0F;
            v.values[64 + i] = 4.0F;
        }
        auto qd = sdxl::cuda::Tensor::from_host_f32(runtime, q);
        auto kd = sdxl::cuda::Tensor::from_host_f32(runtime, k);
        auto vd = sdxl::cuda::Tensor::from_host_f32(runtime, v);
        auto attention = ops.attention(qd, kd, vd, 1, false).to_host_f32(runtime);
        for (float value : attention.values) require_close(value, 3.0F, 0.02F, "attention");

        // Compare the tiled SM80 Tensor Core attention against the original
        // warp-online reference on a nontrivial SDXL-shaped head-dim-64 case.
        constexpr std::size_t attention_sequence = 64;
        constexpr std::size_t attention_heads = 2;
        constexpr std::size_t attention_width = attention_heads * 64;
        std::vector<float> q_values(attention_sequence * attention_width);
        std::vector<float> k_values(attention_sequence * attention_width);
        std::vector<float> v_values(attention_sequence * attention_width);
        for (std::size_t index = 0; index < q_values.size(); ++index) {
            q_values[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.25F;
            k_values[index] = std::cos(static_cast<float>(index) * 0.017F) * 0.25F;
            v_values[index] = std::sin(static_cast<float>(index) * 0.007F);
        }
        const sdxl::FloatTensor q_compare{{1, attention_sequence, attention_width}, q_values};
        const sdxl::FloatTensor k_compare{{1, attention_sequence, attention_width}, k_values};
        const sdxl::FloatTensor v_compare{{1, attention_sequence, attention_width}, v_values};
        sdxl::cuda::RuntimeOptions flash_options;
        flash_options.attention_backend = sdxl::cuda::AttentionBackend::FlashSM80;
        sdxl::cuda::Runtime flash_runtime(flash_options);
        sdxl::cuda::Ops flash_ops(flash_runtime);
        const auto flash_result = flash_ops.attention(
            sdxl::cuda::Tensor::from_host_f32(flash_runtime, q_compare),
            sdxl::cuda::Tensor::from_host_f32(flash_runtime, k_compare),
            sdxl::cuda::Tensor::from_host_f32(flash_runtime, v_compare),
            attention_heads, false).to_host_f32(flash_runtime);

        sdxl::cuda::RuntimeOptions reference_options;
        reference_options.attention_backend = sdxl::cuda::AttentionBackend::WarpOnline;
        sdxl::cuda::Runtime reference_runtime(reference_options);
        sdxl::cuda::Ops reference_ops(reference_runtime);
        const auto reference_result = reference_ops.attention(
            sdxl::cuda::Tensor::from_host_f32(reference_runtime, q_compare),
            sdxl::cuda::Tensor::from_host_f32(reference_runtime, k_compare),
            sdxl::cuda::Tensor::from_host_f32(reference_runtime, v_compare),
            attention_heads, false).to_host_f32(reference_runtime);
        if (flash_result.values.size() != reference_result.values.size()) {
            throw std::runtime_error("Flash attention result shape mismatch");
        }
        for (std::size_t index = 0; index < flash_result.values.size(); ++index) {
            require_close(flash_result.values[index], reference_result.values[index],
                          0.04F, "flash-sm80 attention");
        }

        // Generic FP32 online attention at the real SDXL VAE head width of 512.
        sdxl::FloatTensor q_vae{{1, 2, 512}, std::vector<float>(1024, 0.0F)};
        sdxl::FloatTensor k_vae = q_vae;
        sdxl::FloatTensor v_vae{{1, 2, 512}, std::vector<float>(1024, 0.0F)};
        for (std::size_t i = 0; i < 512; ++i) {
            v_vae.values[i] = 2.0F;
            v_vae.values[512 + i] = 4.0F;
        }
        auto q_vae_device = ops.cast(
            sdxl::cuda::Tensor::from_host_f32(runtime, q_vae), sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto k_vae_device = ops.cast(
            sdxl::cuda::Tensor::from_host_f32(runtime, k_vae), sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto v_vae_device = ops.cast(
            sdxl::cuda::Tensor::from_host_f32(runtime, v_vae), sdxl::cuda::ScalarType::Float32, sdxl::cuda::TensorRole::VAE);
        auto attention_vae = ops.attention(
            q_vae_device, k_vae_device, v_vae_device, 1, false).to_host_f32(runtime);
        for (float value : attention_vae.values) {
            require_close(value, 3.0F, 0.001F, "VAE FP32 attention");
        }

        // Classifier-free guidance kernel.
        auto cfg_device = upload(runtime, {2, 1, 1, 4}, {
            1.0F, 2.0F, 3.0F, 4.0F,
            3.0F, 4.0F, 5.0F, 6.0F});
        auto guided = ops.classifier_free_guidance(
            cfg_device, 1, 2.0F, 0.0F).to_host_f32(runtime);
        require_close(guided.values[0], 5.0F, 0.02F, "cfg");
        require_close(guided.values[3], 8.0F, 0.02F, "cfg");

        // Euler scheduler tensor equation on CUDA.
        auto scheduler_sample = upload(runtime, {1, 1, 1, 2}, {1.0F, 2.0F});
        auto scheduler_prediction = upload(runtime, {1, 1, 1, 2}, {0.5F, -0.5F});
        auto euler = ops.euler_step(
            scheduler_prediction, scheduler_sample, 1.0F, 0.0F, 0).to_host_f32(runtime);
        require_close(euler.values[0], 0.5F, 0.02F, "Euler");
        require_close(euler.values[1], 2.5F, 0.02F, "Euler");

        // Fused Euler input scaling and CFG/scheduler update match the separate path.
        const auto scaled_repeated = ops.euler_scale_repeat_input(
            scheduler_sample, 1.0F, 2).to_host_f32(runtime);
        require_close(scaled_repeated.values[0], 0.7071067F, 0.02F, "Euler scale repeat");
        require_close(scaled_repeated.values[1], 1.4142135F, 0.02F, "Euler scale repeat");
        require_close(scaled_repeated.values[2], 0.7071067F, 0.02F, "Euler scale repeat");
        require_close(scaled_repeated.values[3], 1.4142135F, 0.02F, "Euler scale repeat");

        auto cfg_prediction = upload(runtime, {2, 1, 1, 2}, {
            0.25F, -0.25F,
            0.75F, 0.25F});
        const auto separate_cfg = ops.classifier_free_guidance(
            cfg_prediction, 1, 2.0F, 0.0F);
        const auto separate_euler = ops.euler_step(
            separate_cfg, scheduler_sample, 1.0F, 0.5F, 0).to_host_f32(runtime);
        const auto fused_euler = ops.euler_cfg_step(
            cfg_prediction, scheduler_sample, 1, 2.0F,
            1.0F, 0.5F, 0).to_host_f32(runtime);
        for (std::size_t index = 0; index < fused_euler.values.size(); ++index) {
            require_close(fused_euler.values[index], separate_euler.values[index],
                          0.02F, "fused Euler CFG");
        }

        const auto separate_ddim = ops.ddim_step(
            separate_cfg, scheduler_sample, 0.5F, 0.75F, 0.0F, 0).to_host_f32(runtime);
        const auto fused_ddim = ops.ddim_cfg_step(
            cfg_prediction, scheduler_sample, 1, 2.0F,
            0.5F, 0.75F, 0.0F, 0).to_host_f32(runtime);
        for (std::size_t index = 0; index < fused_ddim.values.size(); ++index) {
            require_close(fused_ddim.values[index], separate_ddim.values[index],
                          0.02F, "fused DDIM CFG");
        }

        // Batch Philox generation uses one persistent-arena seed tensor rather than
        // cudaMalloc/cudaFree per request and remains deterministic per image.
        auto batch_noise_a = sdxl::cuda::Tensor::allocate(
            runtime, {2, 1, 1, 16}, sdxl::cuda::ScalarType::Float16,
            sdxl::cuda::TensorRole::Model);
        auto batch_noise_b = sdxl::cuda::Tensor::allocate(
            runtime, {2, 1, 1, 16}, sdxl::cuda::ScalarType::Float16,
            sdxl::cuda::TensorRole::Model);
        const std::array<std::uint64_t, 2> batch_seeds{{1234ULL, 5678ULL}};
        ops.random_normal_batch_into(batch_noise_a, batch_seeds);
        ops.random_normal_batch_into(batch_noise_b, batch_seeds);
        const auto noise_a = batch_noise_a.to_host_f32(runtime);
        const auto noise_b = batch_noise_b.to_host_f32(runtime);
        bool distinct_batches = false;
        for (std::size_t index = 0; index < noise_a.values.size(); ++index) {
            require_close(noise_a.values[index], noise_b.values[index], 0.0F,
                          "batch Philox determinism");
            if (index < 16 && noise_a.values[index] != noise_a.values[index + 16]) {
                distinct_batches = true;
            }
        }
        if (!distinct_batches) throw std::runtime_error("batch Philox seeds produced identical images");

        // Exact-size arena reuse avoids another driver allocation for the same temporary.
        const auto arena_before = runtime.memory_arena_stats();
        {
            auto temporary = sdxl::cuda::Tensor::allocate(
                runtime, {4096}, sdxl::cuda::ScalarType::Float16,
                sdxl::cuda::TensorRole::Model);
            (void)temporary;
        }
        const auto arena_mid = runtime.memory_arena_stats();
        {
            auto temporary = sdxl::cuda::Tensor::allocate(
                runtime, {4096}, sdxl::cuda::ScalarType::Float16,
                sdxl::cuda::TensorRole::Model);
            (void)temporary;
        }
        const auto arena_after = runtime.memory_arena_stats();
        if (arena_after.cache_hits <= arena_mid.cache_hits ||
            arena_after.driver_allocations != arena_mid.driver_allocations ||
            arena_mid.driver_allocations < arena_before.driver_allocations) {
            throw std::runtime_error("CUDA temporary memory arena did not reuse an exact-size block");
        }

        // The production coalescing slab services differently sized temporary
        // allocations without new CUDA driver allocations after construction.
        sdxl::cuda::RuntimeOptions slab_options;
        slab_options.arena_reserve_bytes = 1U << 20U;
        sdxl::cuda::Runtime slab_runtime(slab_options);
        const auto slab_start = slab_runtime.memory_arena_stats();
        {
            auto first = sdxl::cuda::Tensor::allocate(
                slab_runtime, {8192}, sdxl::cuda::ScalarType::Float16);
            auto second = sdxl::cuda::Tensor::allocate(
                slab_runtime, {4096}, sdxl::cuda::ScalarType::Float16);
            (void)first;
            (void)second;
        }
        const auto slab_end = slab_runtime.memory_arena_stats();
        if (slab_end.slab_suballocations < slab_start.slab_suballocations + 2 ||
            slab_end.driver_allocations != slab_start.driver_allocations ||
            slab_end.slab_free_bytes != slab_end.slab_bytes) {
            throw std::runtime_error("persistent coalescing GPU slab did not serve/recover temporary blocks");
        }

        // Mixed UNet residency policy: rank-2 matrix weights become FP8 while
        // convolutions, normalization parameters, and biases remain FP16.
        sdxl::SDXLModel tiny_model(miniature_config());
        OwnedWeights owned_weights;
        owned_weights.bind_unet(tiny_model);
        sdxl::cuda::WeightStore tiny_weights(runtime, tiny_model);
        const std::filesystem::path fp8_cache_path = "sdxl_cuda_test_unet.sdxlfp8";
        sdxl::cuda::FP8CacheOptions cache_write;
        cache_write.path = fp8_cache_path;
        cache_write.key = "miniature-unet-cache-v2";
        cache_write.read = false;
        cache_write.write = true;
        sdxl::cuda::FP8CacheStats cache_write_stats;
        const auto tiny_unet_stats = tiny_weights.load_unet_fp8(
            {}, &cache_write, &cache_write_stats);
        if (!cache_write_stats.written || !std::filesystem::is_regular_file(fp8_cache_path)) {
            throw std::runtime_error("packed FP8 UNet cache was not written");
        }
        if (tiny_unet_stats.fp8_tensors() == 0 || tiny_unet_stats.fp16_tensors == 0) {
            throw std::runtime_error("mixed FP8/FP16 UNet loading policy was not exercised");
        }
        const auto& packed_weight = tiny_weights.get("unet.time_embedding.linear_1.weight");
        if (packed_weight.type() != sdxl::cuda::ScalarType::Float8E4M3 ||
            tiny_weights.get("unet.conv_in.weight").type() !=
                sdxl::cuda::ScalarType::Float16) {
            throw std::runtime_error("UNet FP8 loading selected the wrong tensor classes");
        }
        const auto expected_layout = runtime.fp8_execution_mode() ==
                sdxl::cuda::FP8ExecutionMode::WeightOnlyTensorCore
            ? sdxl::cuda::FP8StorageLayout::KMajorKN
            : sdxl::cuda::FP8StorageLayout::RowMajorNK;
        if (packed_weight.fp8_storage_layout() != expected_layout) {
            throw std::runtime_error("FP8 weight storage layout does not match the execution backend");
        }
        const auto packed_logical = packed_weight.to_host_f32(runtime);
        for (std::size_t index = 0; index < packed_logical.values.size(); ++index) {
            const float expected = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.03125F;
            require_close(packed_logical.values[index], expected, 0.04F,
                          "tensor-wide K-major FP8 logical order");
        }
        tiny_weights.unload_prefix("unet.");
        sdxl::cuda::FP8CacheOptions cache_read = cache_write;
        cache_read.read = true;
        cache_read.write = false;
        sdxl::cuda::FP8CacheStats cache_read_stats;
        [[maybe_unused]] const auto cached_unet_stats = tiny_weights.load_unet_fp8(
            {}, &cache_read, &cache_read_stats);
        if (!cache_read_stats.hit || cache_read_stats.tensors_loaded == 0) {
            throw std::runtime_error("packed FP8 UNet cache was not reused");
        }
        const auto& cached_packed_weight = tiny_weights.get("unet.time_embedding.linear_1.weight");
        if (cached_packed_weight.fp8_storage_layout() != expected_layout) {
            throw std::runtime_error("FP8 cache did not preserve the backend-specific packed layout");
        }
        const auto cached_logical = cached_packed_weight.to_host_f32(runtime);
        for (std::size_t index = 0; index < cached_logical.values.size(); ++index) {
            const float expected = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.03125F;
            require_close(cached_logical.values[index], expected, 0.04F,
                          "cached tensor-wide K-major FP8 logical order");
        }
        tiny_weights.unload_prefix("unet.");
        std::filesystem::remove(fp8_cache_path);
        sdxl::cuda::FP8WeightLoadOptions e5_channel_options;
        e5_channel_options.format = sdxl::cuda::FP8Format::E5M2;
        e5_channel_options.scale_mode = sdxl::cuda::FP8ScaleMode::PerOutputChannel;
        e5_channel_options.backend = sdxl::cuda::FP8Backend::WeightOnly;
        const auto e5_channel_stats = tiny_weights.load_unet_fp8(e5_channel_options);
        const auto& channel_weight = tiny_weights.get("unet.time_embedding.linear_1.weight");
        if (e5_channel_stats.fp8_e5m2_tensors == 0 ||
            e5_channel_stats.channel_scaled_tensors == 0 ||
            channel_weight.type() != sdxl::cuda::ScalarType::Float8E5M2 ||
            channel_weight.dequant_scale_mode() != sdxl::cuda::FP8ScaleMode::PerOutputChannel) {
            throw std::runtime_error("E5M2 per-channel UNet runtime was not exercised");
        }
        tiny_weights.unload_prefix("unet.");

        // Warm and replay a reusable fixed-shape denoising CUDA Graph. The first
        // eager pass populates cuBLASLt/cuDNN plans and persistent workspaces.
        [[maybe_unused]] const auto graph_unet_stats = tiny_weights.load_unet_fp8();
        sdxl::cuda::ClassifierFreeConditioning tiny_conditioning;
        tiny_conditioning.batch_size = 2;
        tiny_conditioning.prompt_embeds = upload(
            runtime, {4, 4, 8}, std::vector<float>(4 * 4 * 8, 0.0F));
        tiny_conditioning.pooled_text_embeds = upload(
            runtime, {4, 8}, std::vector<float>(4 * 8, 0.0F));
        sdxl::cuda::DenoiseOptions tiny_denoise_options;
        tiny_denoise_options.width = 8;
        tiny_denoise_options.height = 8;
        tiny_denoise_options.inference_steps = 1;
        tiny_denoise_options.guidance_scale = 1.0F;
        tiny_denoise_options.seed = 11;
        tiny_denoise_options.batch_seeds = {11, 12};
        {
            sdxl::cuda::Denoiser warm_denoiser(
                runtime, tiny_model, tiny_weights);
            auto warm_result = warm_denoiser.denoise(
                tiny_conditioning, tiny_denoise_options);
            runtime.synchronize();
            if (warm_result.latents.shape() !=
                std::vector<std::size_t>{2, 4, 1, 1}) {
                throw std::runtime_error("miniature eager denoise output shape mismatch");
            }

            sdxl::cuda::DenoiseGraph graph(
                runtime, tiny_model, tiny_weights,
                tiny_conditioning, tiny_denoise_options);
            const std::array<std::uint64_t, 2> graph_seeds{{13ULL, 14ULL}};
            auto graph_result = graph.launch(tiny_conditioning, graph_seeds);
            const auto graph_host = graph_result.latents.to_host_f32(runtime);
            if (graph_host.shape != std::vector<std::size_t>{2, 4, 1, 1}) {
                throw std::runtime_error("miniature CUDA Graph output shape mismatch");
            }
            for (const float value : graph_host.values) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("miniature CUDA Graph produced non-finite latents");
                }
            }

            // Guidance scale 1 follows the production no-CFG fast path: only
            // positive conditioning is encoded and the UNet batch is not
            // doubled. This is the important fast path for Lightning/Turbo
            // checkpoints commonly run with cfg=1.
            sdxl::cuda::ClassifierFreeConditioning single_conditioning;
            single_conditioning.batch_size = 2;
            single_conditioning.classifier_free = false;
            single_conditioning.prompt_embeds = upload(
                runtime, {2, 4, 8}, std::vector<float>(2 * 4 * 8, 0.0F));
            single_conditioning.pooled_text_embeds = upload(
                runtime, {2, 8}, std::vector<float>(2 * 8, 0.0F));
            auto single_result = warm_denoiser.denoise(
                single_conditioning, tiny_denoise_options);
            const auto single_host = single_result.latents.to_host_f32(runtime);
            if (single_host.shape != std::vector<std::size_t>{2, 4, 1, 1}) {
                throw std::runtime_error("CFG-bypass denoise output shape mismatch");
            }
            for (const float value : single_host.values) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("CFG-bypass denoise produced non-finite latents");
                }
            }
        }
        tiny_weights.unload_prefix("unet.");

        // Complete CUDA VAE decoder and RGB8/PNG image conversion on a miniature model.
        owned_weights.bind_vae(tiny_model);
        const auto tiny_stats = tiny_weights.load_prefix(
            "vae.", sdxl::cuda::ScalarType::Float32);
        if (tiny_stats.tensors == 0) throw std::runtime_error("miniature VAE uploaded no weights");
        sdxl::cuda::VAEOptions tiny_vae_options;
        tiny_vae_options.norm_groups = 32;
        tiny_vae_options.scaling_factor = 0.13025F;
        sdxl::cuda::VAE tiny_vae(runtime, tiny_model, tiny_weights, tiny_vae_options);
        auto tiny_latents = upload(runtime, {1, 4, 1, 1}, {0.0F, 0.0F, 0.0F, 0.0F});
        auto decoded = tiny_vae.decode(tiny_latents);
        if (decoded.shape() != std::vector<std::size_t>{1, 3, 8, 8}) {
            throw std::runtime_error("miniature CUDA VAE output shape mismatch");
        }
        sdxl::cuda::ImageConverter converter(runtime);
        const auto images = converter.to_rgb8(decoded);
        if (images.size() != 1 || images[0].pixels.size() != 8 * 8 * 3) {
            throw std::runtime_error("CUDA RGB conversion shape mismatch");
        }
        for (const std::uint8_t value : images[0].pixels) {
            if (value != 128U) throw std::runtime_error("CUDA RGB conversion value mismatch");
        }
        const std::filesystem::path png_path = "sdxl_cuda_test_output.png";
        sdxl::cuda::write_png(png_path, images[0]);
        if (!std::filesystem::is_regular_file(png_path) ||
            std::filesystem::file_size(png_path) <= 32) {
            throw std::runtime_error("PNG writer did not create a valid-sized file");
        }
        std::filesystem::remove(png_path);

        std::cout << "CUDA FP8, operator, CFG, VAE, and image tests passed on "
                  << runtime.device_properties().name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
