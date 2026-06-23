#pragma once

#include "sdxl/cuda/tensor.hpp"
#include "sdxl/cuda/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace sdxl::cuda {

enum class LinearActivation : std::uint8_t { None, SiLU, GELU, QuickGELU, GEGLU };

class Ops final {
public:
    explicit Ops(const Runtime& runtime);

    [[nodiscard]] Tensor linear(const Tensor& input,
                                const Tensor& weight,
                                const Tensor* bias = nullptr) const;
    [[nodiscard]] Tensor linear_activation(const Tensor& input,
                                           const Tensor& weight,
                                           const Tensor& bias,
                                           LinearActivation activation) const;
    [[nodiscard]] Tensor quantize_fp8(const Tensor& input, ScalarType fp8_type) const;
    [[nodiscard]] Tensor quantize_e4m3(const Tensor& input) const;
    [[nodiscard]] Tensor quantize_e5m2(const Tensor& input) const;
    [[nodiscard]] Tensor convolution_nchw(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor* bias,
                                         int stride_y,
                                         int stride_x,
                                         int pad_y,
                                         int pad_x) const;

    [[nodiscard]] Tensor layer_norm(const Tensor& input,
                                    const Tensor& weight,
                                    const Tensor& bias,
                                    float epsilon = 1.0e-5F) const;
    [[nodiscard]] Tensor group_norm_nchw(const Tensor& input,
                                        const Tensor& weight,
                                        const Tensor& bias,
                                        std::size_t groups,
                                        float epsilon = 1.0e-5F) const;
    [[nodiscard]] Tensor group_norm_silu_nchw(const Tensor& input,
                                             const Tensor& weight,
                                             const Tensor& bias,
                                             std::size_t groups,
                                             float epsilon = 1.0e-5F) const;

    [[nodiscard]] Tensor cast(const Tensor& input, ScalarType destination_type,
                              TensorRole destination_role = TensorRole::Model) const;
    [[nodiscard]] Tensor cast_scale(const Tensor& input, ScalarType destination_type,
                                    TensorRole destination_role, float scale) const;
    [[nodiscard]] Tensor add(const Tensor& first, const Tensor& second) const;
    [[nodiscard]] Tensor add_silu(const Tensor& first, const Tensor& second) const;
    void add_in_place(Tensor& destination, const Tensor& source) const;
    void add_last_dim_bias_in_place(Tensor& destination, const Tensor& bias) const;
    void bias_activation_in_place(Tensor& destination, const Tensor& bias,
                                  LinearActivation activation) const;
    [[nodiscard]] Tensor bias_geglu(const Tensor& projected, const Tensor& bias) const;
    [[nodiscard]] Tensor scale(const Tensor& tensor, float scale) const;
    void scale_in_place(Tensor& tensor, float scale) const;
    [[nodiscard]] Tensor add_spatial_bias(const Tensor& nchw, const Tensor& batch_channels) const;
    [[nodiscard]] Tensor silu(const Tensor& input) const;
    void silu_in_place(Tensor& input) const;
    [[nodiscard]] Tensor quick_gelu(const Tensor& input) const;
    [[nodiscard]] Tensor gelu(const Tensor& input) const;
    [[nodiscard]] Tensor geglu(const Tensor& input) const;

    [[nodiscard]] Tensor flatten_spatial(const Tensor& nchw) const;
    [[nodiscard]] Tensor unflatten_spatial(const Tensor& bsc,
                                           std::size_t height,
                                           std::size_t width) const;
    [[nodiscard]] Tensor concat_channels(const Tensor& first, const Tensor& second) const;
    [[nodiscard]] Tensor concat_last_dim(const Tensor& first, const Tensor& second) const;
    [[nodiscard]] Tensor concat_batch(const Tensor& first, const Tensor& second) const;
    [[nodiscard]] Tensor repeat_batch(const Tensor& input, std::size_t repeats) const;
    [[nodiscard]] Tensor nearest_upsample(const Tensor& input,
                                         std::size_t target_height,
                                         std::size_t target_width) const;

    [[nodiscard]] Tensor embedding(const Tensor& token_ids,
                                   const Tensor& token_embedding,
                                   const Tensor& position_embedding) const;
    [[nodiscard]] Tensor pool_eos(const Tensor& hidden,
                                  const Tensor& token_ids) const;

    [[nodiscard]] Tensor attention(const Tensor& query,
                                   const Tensor& key,
                                   const Tensor& value,
                                   std::size_t heads,
                                   bool causal = false,
                                   const Tensor* key_mask = nullptr) const;

    [[nodiscard]] Tensor timestep_embedding_scalar(float timestep,
                                                   std::size_t batch,
                                                   std::size_t dimension) const;
    [[nodiscard]] Tensor timestep_embedding_values(const Tensor& values,
                                                   std::size_t dimension) const;

    [[nodiscard]] Tensor classifier_free_guidance(const Tensor& model_output,
                                                  std::size_t batch,
                                                  float guidance_scale,
                                                  float guidance_rescale) const;
    [[nodiscard]] Tensor euler_scale_input(const Tensor& sample, float sigma) const;
    [[nodiscard]] Tensor euler_scale_repeat_input(const Tensor& sample,
                                                  float sigma,
                                                  std::size_t repeats) const;
    [[nodiscard]] Tensor predicted_original(const Tensor& model_output,
                                             const Tensor& sample,
                                             float sigma,
                                             int prediction_type) const;
    [[nodiscard]] Tensor batch_slice(const Tensor& input,
                                     std::size_t start_batch,
                                     std::size_t batch_count) const;
    [[nodiscard]] Tensor combine(const Tensor& a, float ca,
                                 const Tensor* b = nullptr, float cb = 0.0F,
                                 const Tensor* c = nullptr, float cc = 0.0F,
                                 const Tensor* noise = nullptr, float cn = 0.0F) const;
    [[nodiscard]] Tensor dpmpp_2m_step(const Tensor& denoised,
                                       const Tensor& sample,
                                       const Tensor* old_denoised,
                                       float sigma,
                                       float sigma_next,
                                       float sigma_previous = 0.0F) const;
    [[nodiscard]] Tensor euler_step(const Tensor& model_output,
                                    const Tensor& sample,
                                    float sigma,
                                    float sigma_next,
                                    int prediction_type) const;
    [[nodiscard]] Tensor euler_cfg_step(const Tensor& model_output,
                                        const Tensor& sample,
                                        std::size_t batch,
                                        float guidance_scale,
                                        float sigma,
                                        float sigma_next,
                                        int prediction_type) const;
    [[nodiscard]] Tensor ddim_step(const Tensor& model_output,
                                   const Tensor& sample,
                                   float alpha_t,
                                   float alpha_prev,
                                   float eta,
                                   int prediction_type,
                                   const Tensor* noise = nullptr) const;
    [[nodiscard]] Tensor ddim_cfg_step(const Tensor& model_output,
                                       const Tensor& sample,
                                       std::size_t batch,
                                       float guidance_scale,
                                       float alpha_t,
                                       float alpha_prev,
                                       float eta,
                                       int prediction_type,
                                       const Tensor* noise = nullptr) const;

    [[nodiscard]] Tensor random_normal(std::vector<std::size_t> shape,
                                       std::uint64_t seed,
                                       float scale = 1.0F,
                                       ScalarType type = ScalarType::Float16,
                                       TensorRole role = TensorRole::Model) const;
    void random_normal_into(Tensor& output,
                            std::uint64_t seed,
                            float scale = 1.0F) const;
    void random_normal_batch_into(Tensor& output,
                                  std::span<const std::uint64_t> seeds,
                                  float scale = 1.0F) const;
    void check_finite(const Tensor& tensor, std::string_view name) const;

private:
    const Runtime* runtime_ = nullptr;
};

} // namespace sdxl::cuda
