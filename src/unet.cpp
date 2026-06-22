#include "sdxl/unet.hpp"

#include "sdxl/model.hpp"
#include "sdxl/safetensors.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

namespace sdxl {

FloatTensor zeros(std::vector<std::size_t> shape);

namespace {

struct SDXLUNetInput {
    const FloatTensor& latents;
    const FloatTensor& encoder_hidden_states;
    const FloatTensor& pooled_text_embeds;
    const FloatTensor& add_time_ids;
    float timestep = 0.0F;
};

class TensorReader final {
public:
    explicit TensorReader(const TensorView& view) : view_(view) {
        if (view_.data == nullptr) throw Error("bound tensor has a null data pointer: " + view_.source_key);
        if (view_.shape.size() != view_.strides_bytes.size()) {
            throw Error("bound tensor has an invalid stride rank: " + view_.source_key);
        }
    }

    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return view_.shape; }

    [[nodiscard]] float at1(std::size_t a) const { return read(index_offset({a})); }
    [[nodiscard]] float at2(std::size_t a, std::size_t b) const { return read(index_offset({a, b})); }
    [[nodiscard]] float at4(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const { return read(index_offset({a, b, c, d})); }

private:
    [[nodiscard]] const std::byte* index_offset(std::initializer_list<std::size_t> index) const {
        if (index.size() != view_.shape.size()) throw Error("tensor rank mismatch");
        std::ptrdiff_t offset = 0;
        std::size_t dim = 0;
        for (const std::size_t value : index) {
            if (value >= view_.shape[dim]) throw Error("tensor index out of range");
            offset += static_cast<std::ptrdiff_t>(value) * view_.strides_bytes[dim];
            ++dim;
        }
        return view_.data + offset;
    }

    [[nodiscard]] static float half_to_float(std::uint16_t bits) noexcept {
        const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
        const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
        std::uint32_t mantissa = bits & 0x03FFU;
        std::uint32_t output = 0;
        if (exponent == 0) {
            if (mantissa == 0) {
                output = sign;
            } else {
                int unbiased_exponent = -14;
                while ((mantissa & 0x0400U) == 0) {
                    mantissa <<= 1U;
                    --unbiased_exponent;
                }
                mantissa &= 0x03FFU;
                const auto exp32 = static_cast<std::uint32_t>(unbiased_exponent + 127);
                output = sign | (exp32 << 23U) | (mantissa << 13U);
            }
        } else if (exponent == 0x1FU) {
            output = sign | 0x7F800000U | (mantissa << 13U);
        } else {
            const std::uint32_t exp32 = exponent + (127U - 15U);
            output = sign | (exp32 << 23U) | (mantissa << 13U);
        }
        return std::bit_cast<float>(output);
    }

    [[nodiscard]] static float bf16_to_float(std::uint16_t bits) noexcept {
        return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }

    [[nodiscard]] static float fp8_to_float(std::uint8_t bits,
                                            unsigned exponent_bits,
                                            unsigned mantissa_bits,
                                            int exponent_bias,
                                            bool finite_only) noexcept {
        const bool negative = (bits & 0x80U) != 0;
        const unsigned exponent_mask = (1U << exponent_bits) - 1U;
        const unsigned mantissa_mask = (1U << mantissa_bits) - 1U;
        const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
        const unsigned mantissa = bits & mantissa_mask;
        float value = 0.0F;
        if (exponent == 0) {
            if (mantissa != 0) {
                value = std::ldexp(static_cast<float>(mantissa) /
                                       static_cast<float>(1U << mantissa_bits),
                                   1 - exponent_bias);
            }
        } else if (exponent == exponent_mask && !finite_only) {
            value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
        } else if (finite_only && exponent == exponent_mask && mantissa == mantissa_mask) {
            value = std::numeric_limits<float>::quiet_NaN();
        } else {
            value = std::ldexp(1.0F + static_cast<float>(mantissa) /
                                          static_cast<float>(1U << mantissa_bits),
                               static_cast<int>(exponent) - exponent_bias);
        }
        return negative ? -value : value;
    }

    [[nodiscard]] static float read(const std::byte* address, DType dtype) {
        switch (dtype) {
        case DType::F16: {
            std::uint16_t bits = 0; std::memcpy(&bits, address, sizeof(bits)); return half_to_float(bits);
        }
        case DType::BF16: {
            std::uint16_t bits = 0; std::memcpy(&bits, address, sizeof(bits)); return bf16_to_float(bits);
        }
        case DType::F32: {
            float value = 0.0F; std::memcpy(&value, address, sizeof(value)); return value;
        }
        case DType::F64: {
            double value = 0.0; std::memcpy(&value, address, sizeof(value)); return static_cast<float>(value);
        }
        case DType::F8E4M3FN:
            return fp8_to_float(std::to_integer<std::uint8_t>(*address), 4, 3, 7, true);
        case DType::F8E5M2:
            return fp8_to_float(std::to_integer<std::uint8_t>(*address), 5, 2, 15, false);
        default:
            throw Error("UNet requires floating-point weights");
        }
    }

    [[nodiscard]] float read(const std::byte* address) const { return read(address, view_.dtype); }

    const TensorView& view_;
};

[[nodiscard]] const TensorView& require_tensor(const SDXLModel& model, const std::string& logical_name) {
    const ParameterSlot* slot = model.graph().find_parameter(logical_name);
    if (slot == nullptr) throw Error("UNet parameter is not declared: " + logical_name);
    if (!slot->tensor.has_value()) throw Error("UNet parameter is not loaded: " + logical_name);
    if (slot->tensor->shape != slot->expected_shape) {
        throw Error("UNet parameter has wrong shape: " + logical_name + " expected " +
                    shape_to_string(slot->expected_shape) + " loaded " +
                    shape_to_string(slot->tensor->shape));
    }
    return *slot->tensor;
}

[[nodiscard]] std::size_t product(const std::vector<std::size_t>& shape) {
    std::size_t total = 1;
    for (const std::size_t dim : shape) total *= dim;
    return total;
}

[[nodiscard]] FloatTensor make_tensor(std::vector<std::size_t> shape, std::vector<float> values) {
    if (product(shape) != values.size()) throw Error("FloatTensor shape does not match storage");
    return FloatTensor{std::move(shape), std::move(values)};
}

[[nodiscard]] std::size_t offset2(const FloatTensor& tensor, std::size_t a, std::size_t b) {
    return a * tensor.shape[1] + b;
}
[[nodiscard]] std::size_t offset3(const FloatTensor& tensor, std::size_t a, std::size_t b, std::size_t c) {
    return (a * tensor.shape[1] + b) * tensor.shape[2] + c;
}
[[nodiscard]] std::size_t offset4(const FloatTensor& tensor, std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return ((a * tensor.shape[1] + b) * tensor.shape[2] + c) * tensor.shape[3] + d;
}

void add_in_place(FloatTensor& a, const FloatTensor& b) {
    if (a.shape != b.shape) throw Error("tensor shape mismatch in add_in_place");
    for (std::size_t i = 0; i < a.values.size(); ++i) a.values[i] += b.values[i];
}

void silu_in_place(FloatTensor& tensor) {
    for (float& value : tensor.values) value = value / (1.0F + std::exp(-value));
}

[[nodiscard]] float gelu(float x) noexcept {
    constexpr float inv = 0.7071067811865475244F;
    return 0.5F * x * (1.0F + std::erf(x * inv));
}

void check_finite(const FloatTensor& tensor, std::string_view name) {
    for (float value : tensor.values) {
        if (!std::isfinite(value)) throw Error(std::string(name) + " contains NaN or infinity");
    }
}

[[nodiscard]] FloatTensor linear_2d(const FloatTensor& input,
                                    const TensorView& weight_view,
                                    const TensorView* bias_view = nullptr) {
    if (input.shape.size() != 2) throw Error("linear_2d expects [rows,in_features]");
    const std::size_t rows = input.shape[0];
    const std::size_t in_features = input.shape[1];
    const TensorReader weight(weight_view);
    if (weight.shape().size() != 2 || weight.shape()[1] != in_features) throw Error("linear_2d weight mismatch");
    const std::size_t out_features = static_cast<std::size_t>(weight.shape()[0]);
    std::unique_ptr<TensorReader> bias;
    if (bias_view != nullptr) bias = std::make_unique<TensorReader>(*bias_view);
    FloatTensor out = zeros({rows, out_features});
    for (std::size_t r = 0; r < rows; ++r) {
        const float* src = input.values.data() + r * in_features;
        for (std::size_t o = 0; o < out_features; ++o) {
            float v = bias ? bias->at1(o) : 0.0F;
            for (std::size_t i = 0; i < in_features; ++i) v += src[i] * weight.at2(o, i);
            out.values[offset2(out, r, o)] = v;
        }
    }
    return out;
}

[[nodiscard]] FloatTensor linear_3d(const FloatTensor& input,
                                    const TensorView& weight_view,
                                    const TensorView* bias_view = nullptr) {
    if (input.shape.size() != 3) throw Error("linear_3d expects [batch,seq,features]");
    const std::size_t batch = input.shape[0];
    const std::size_t seq = input.shape[1];
    const std::size_t features = input.shape[2];
    FloatTensor flat = make_tensor({batch * seq, features}, input.values);
    FloatTensor projected = linear_2d(flat, weight_view, bias_view);
    return make_tensor({batch, seq, projected.shape[1]}, std::move(projected.values));
}

[[nodiscard]] FloatTensor layer_norm_last_dim(const FloatTensor& input,
                                              const TensorView& weight_view,
                                              const TensorView& bias_view,
                                              float epsilon = 1.0e-5F) {
    if (input.shape.empty()) throw Error("layer_norm expects rank > 0");
    const std::size_t width = input.shape.back();
    const std::size_t rows = input.values.size() / width;
    const TensorReader weight(weight_view);
    const TensorReader bias(bias_view);
    if (weight.shape() != std::vector<std::uint64_t>{width} || bias.shape() != std::vector<std::uint64_t>{width}) {
        throw Error("layer_norm parameter mismatch");
    }
    FloatTensor out = zeros(input.shape);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* src = input.values.data() + row * width;
        float mean = 0.0F;
        for (std::size_t i = 0; i < width; ++i) mean += src[i];
        mean /= static_cast<float>(width);
        float variance = 0.0F;
        for (std::size_t i = 0; i < width; ++i) { const float d = src[i] - mean; variance += d * d; }
        variance /= static_cast<float>(width);
        const float inv = 1.0F / std::sqrt(variance + epsilon);
        float* dst = out.values.data() + row * width;
        for (std::size_t i = 0; i < width; ++i) dst[i] = (src[i] - mean) * inv * weight.at1(i) + bias.at1(i);
    }
    return out;
}

[[nodiscard]] FloatTensor group_norm_2d(const FloatTensor& input,
                                        const TensorView& weight_view,
                                        const TensorView& bias_view,
                                        std::size_t groups,
                                        float epsilon = 1.0e-5F) {
    if (input.shape.size() != 4) throw Error("group_norm_2d expects [B,C,H,W]");
    const std::size_t batch = input.shape[0];
    const std::size_t channels = input.shape[1];
    const std::size_t height = input.shape[2];
    const std::size_t width = input.shape[3];
    if (channels % groups != 0) throw Error("group_norm channels must divide groups");
    const std::size_t cpg = channels / groups;
    const TensorReader weight(weight_view);
    const TensorReader bias(bias_view);
    FloatTensor out = zeros(input.shape);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t g = 0; g < groups; ++g) {
            float mean = 0.0F;
            std::size_t count = 0;
            for (std::size_t c = 0; c < cpg; ++c) {
                const std::size_t channel = g * cpg + c;
                for (std::size_t y = 0; y < height; ++y) {
                    for (std::size_t x = 0; x < width; ++x) { mean += input.values[offset4(input, b, channel, y, x)]; ++count; }
                }
            }
            mean /= static_cast<float>(count);
            float variance = 0.0F;
            for (std::size_t c = 0; c < cpg; ++c) {
                const std::size_t channel = g * cpg + c;
                for (std::size_t y = 0; y < height; ++y) {
                    for (std::size_t x = 0; x < width; ++x) {
                        const float d = input.values[offset4(input, b, channel, y, x)] - mean;
                        variance += d * d;
                    }
                }
            }
            variance /= static_cast<float>(count);
            const float inv = 1.0F / std::sqrt(variance + epsilon);
            for (std::size_t c = 0; c < cpg; ++c) {
                const std::size_t channel = g * cpg + c;
                const float gamma = weight.at1(channel);
                const float beta = bias.at1(channel);
                for (std::size_t y = 0; y < height; ++y) {
                    for (std::size_t x = 0; x < width; ++x) {
                        const float v = input.values[offset4(input, b, channel, y, x)];
                        out.values[offset4(out, b, channel, y, x)] = (v - mean) * inv * gamma + beta;
                    }
                }
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor conv2d(const FloatTensor& input,
                                 const TensorView& weight_view,
                                 const TensorView* bias_view,
                                 std::size_t stride,
                                 std::size_t padding) {
    if (input.shape.size() != 4) throw Error("conv2d expects [B,C,H,W]");
    const std::size_t batch = input.shape[0];
    const std::size_t in_channels = input.shape[1];
    const std::size_t in_height = input.shape[2];
    const std::size_t in_width = input.shape[3];
    const TensorReader weight(weight_view);
    if (weight.shape().size() != 4 || weight.shape()[1] != in_channels || weight.shape()[2] != weight.shape()[3]) {
        throw Error("conv2d weight shape mismatch: " + weight_view.source_key);
    }
    const std::size_t out_channels = static_cast<std::size_t>(weight.shape()[0]);
    const std::size_t kernel = static_cast<std::size_t>(weight.shape()[2]);
    std::unique_ptr<TensorReader> bias;
    if (bias_view != nullptr) bias = std::make_unique<TensorReader>(*bias_view);
    const std::size_t out_height = (in_height + 2 * padding - kernel) / stride + 1;
    const std::size_t out_width = (in_width + 2 * padding - kernel) / stride + 1;
    FloatTensor out = zeros({batch, out_channels, out_height, out_width});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t oy = 0; oy < out_height; ++oy) {
                for (std::size_t ox = 0; ox < out_width; ++ox) {
                    float v = bias ? bias->at1(oc) : 0.0F;
                    for (std::size_t ic = 0; ic < in_channels; ++ic) {
                        for (std::size_t ky = 0; ky < kernel; ++ky) {
                            const std::ptrdiff_t iy = static_cast<std::ptrdiff_t>(oy * stride + ky) - static_cast<std::ptrdiff_t>(padding);
                            if (iy < 0 || iy >= static_cast<std::ptrdiff_t>(in_height)) continue;
                            for (std::size_t kx = 0; kx < kernel; ++kx) {
                                const std::ptrdiff_t ix = static_cast<std::ptrdiff_t>(ox * stride + kx) - static_cast<std::ptrdiff_t>(padding);
                                if (ix < 0 || ix >= static_cast<std::ptrdiff_t>(in_width)) continue;
                                v += input.values[offset4(input, b, ic, static_cast<std::size_t>(iy), static_cast<std::size_t>(ix))] *
                                     weight.at4(oc, ic, ky, kx);
                            }
                        }
                    }
                    out.values[offset4(out, b, oc, oy, ox)] = v;
                }
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor nearest_upsample(const FloatTensor& input,
                                           std::size_t target_height,
                                           std::size_t target_width) {
    if (input.shape.size() != 4) throw Error("nearest_upsample expects [B,C,H,W]");
    if (target_height == 0 || target_width == 0) throw Error("nearest upsample target cannot be zero");
    const std::size_t batch = input.shape[0];
    const std::size_t channels = input.shape[1];
    const std::size_t height = input.shape[2];
    const std::size_t width = input.shape[3];
    FloatTensor out = zeros({batch, channels, target_height, target_width});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t y = 0; y < target_height; ++y) {
                const std::size_t source_y = std::min(
                    height - 1, y * height / target_height);
                for (std::size_t x = 0; x < target_width; ++x) {
                    const std::size_t source_x = std::min(
                        width - 1, x * width / target_width);
                    out.values[offset4(out, b, c, y, x)] =
                        input.values[offset4(input, b, c, source_y, source_x)];
                }
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor concat_channels(const FloatTensor& a, const FloatTensor& b) {
    if (a.shape.size() != 4 || b.shape.size() != 4 || a.shape[0] != b.shape[0] || a.shape[2] != b.shape[2] || a.shape[3] != b.shape[3]) {
        throw Error("concat_channels expects matching [B,*,H,W]");
    }
    FloatTensor out = zeros({a.shape[0], a.shape[1] + b.shape[1], a.shape[2], a.shape[3]});
    for (std::size_t batch = 0; batch < out.shape[0]; ++batch) {
        for (std::size_t c = 0; c < a.shape[1]; ++c) {
            for (std::size_t y = 0; y < out.shape[2]; ++y) {
                for (std::size_t x = 0; x < out.shape[3]; ++x) out.values[offset4(out, batch, c, y, x)] = a.values[offset4(a, batch, c, y, x)];
            }
        }
        for (std::size_t c = 0; c < b.shape[1]; ++c) {
            for (std::size_t y = 0; y < out.shape[2]; ++y) {
                for (std::size_t x = 0; x < out.shape[3]; ++x) out.values[offset4(out, batch, a.shape[1] + c, y, x)] = b.values[offset4(b, batch, c, y, x)];
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor flatten_spatial(const FloatTensor& input) {
    if (input.shape.size() != 4) throw Error("flatten_spatial expects [B,C,H,W]");
    FloatTensor out = zeros({input.shape[0], input.shape[2] * input.shape[3], input.shape[1]});
    for (std::size_t b = 0; b < input.shape[0]; ++b) {
        for (std::size_t y = 0; y < input.shape[2]; ++y) {
            for (std::size_t x = 0; x < input.shape[3]; ++x) {
                const std::size_t s = y * input.shape[3] + x;
                for (std::size_t c = 0; c < input.shape[1]; ++c) out.values[offset3(out, b, s, c)] = input.values[offset4(input, b, c, y, x)];
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor unflatten_spatial(const FloatTensor& input, std::size_t height, std::size_t width) {
    if (input.shape.size() != 3 || input.shape[1] != height * width) throw Error("unflatten_spatial shape mismatch");
    FloatTensor out = zeros({input.shape[0], input.shape[2], height, width});
    for (std::size_t b = 0; b < out.shape[0]; ++b) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t s = y * width + x;
                for (std::size_t c = 0; c < out.shape[1]; ++c) out.values[offset4(out, b, c, y, x)] = input.values[offset3(input, b, s, c)];
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor full_attention(const FloatTensor& query,
                                         const FloatTensor& key,
                                         const FloatTensor& value,
                                         std::size_t heads) {
    if (query.shape.size() != 3 || key.shape.size() != 3 || value.shape.size() != 3) throw Error("attention expects rank 3");
    const std::size_t batch = query.shape[0];
    const std::size_t q_seq = query.shape[1];
    const std::size_t k_seq = key.shape[1];
    const std::size_t width = query.shape[2];
    if (key.shape[0] != batch || value.shape[0] != batch || key.shape[2] != width || value.shape[2] != width) throw Error("attention width mismatch");
    if (width % heads != 0) throw Error("attention width must divide heads");
    const std::size_t head_dim = width / heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    FloatTensor out = zeros({batch, q_seq, width});
    std::vector<float> score(k_seq);
    std::vector<float> prob(k_seq);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t base = h * head_dim;
            for (std::size_t qi = 0; qi < q_seq; ++qi) {
                float maxv = -std::numeric_limits<float>::infinity();
                for (std::size_t ki = 0; ki < k_seq; ++ki) {
                    float s = 0.0F;
                    for (std::size_t d = 0; d < head_dim; ++d) s += query.values[offset3(query, b, qi, base + d)] * key.values[offset3(key, b, ki, base + d)];
                    s *= scale;
                    score[ki] = s;
                    maxv = std::max(maxv, s);
                }
                float denom = 0.0F;
                for (std::size_t ki = 0; ki < k_seq; ++ki) { prob[ki] = std::exp(score[ki] - maxv); denom += prob[ki]; }
                denom = 1.0F / denom;
                for (std::size_t ki = 0; ki < k_seq; ++ki) prob[ki] *= denom;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    float v = 0.0F;
                    for (std::size_t ki = 0; ki < k_seq; ++ki) v += prob[ki] * value.values[offset3(value, b, ki, base + d)];
                    out.values[offset3(out, b, qi, base + d)] = v;
                }
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor geglu(const FloatTensor& input) {
    if (input.shape.size() != 3 || input.shape[2] % 2 != 0) throw Error("GEGLU width must be even");
    const std::size_t width = input.shape[2] / 2;
    FloatTensor out = zeros({input.shape[0], input.shape[1], width});
    for (std::size_t b = 0; b < input.shape[0]; ++b) {
        for (std::size_t s = 0; s < input.shape[1]; ++s) {
            for (std::size_t i = 0; i < width; ++i) out.values[offset3(out, b, s, i)] = input.values[offset3(input, b, s, i)] * gelu(input.values[offset3(input, b, s, width + i)]);
        }
    }
    return out;
}

[[nodiscard]] FloatTensor transformer_block(const SDXLModel& model,
                                            const std::string& prefix,
                                            const FloatTensor& input,
                                            const FloatTensor& encoder_hidden_states,
                                            std::size_t heads) {
    FloatTensor hidden = input;
    FloatTensor norm1 = layer_norm_last_dim(hidden, require_tensor(model, prefix + ".norm1.weight"), require_tensor(model, prefix + ".norm1.bias"));
    FloatTensor q1 = linear_3d(norm1, require_tensor(model, prefix + ".attn1.to_q.weight"));
    FloatTensor k1 = linear_3d(norm1, require_tensor(model, prefix + ".attn1.to_k.weight"));
    FloatTensor v1 = linear_3d(norm1, require_tensor(model, prefix + ".attn1.to_v.weight"));
    FloatTensor attn1 = full_attention(q1, k1, v1, heads);
    attn1 = linear_3d(attn1, require_tensor(model, prefix + ".attn1.to_out.0.weight"), &require_tensor(model, prefix + ".attn1.to_out.0.bias"));
    add_in_place(hidden, attn1);

    FloatTensor norm2 = layer_norm_last_dim(hidden, require_tensor(model, prefix + ".norm2.weight"), require_tensor(model, prefix + ".norm2.bias"));
    FloatTensor q2 = linear_3d(norm2, require_tensor(model, prefix + ".attn2.to_q.weight"));
    FloatTensor k2 = linear_3d(encoder_hidden_states, require_tensor(model, prefix + ".attn2.to_k.weight"));
    FloatTensor v2 = linear_3d(encoder_hidden_states, require_tensor(model, prefix + ".attn2.to_v.weight"));
    FloatTensor attn2 = full_attention(q2, k2, v2, heads);
    attn2 = linear_3d(attn2, require_tensor(model, prefix + ".attn2.to_out.0.weight"), &require_tensor(model, prefix + ".attn2.to_out.0.bias"));
    add_in_place(hidden, attn2);

    FloatTensor norm3 = layer_norm_last_dim(hidden, require_tensor(model, prefix + ".norm3.weight"), require_tensor(model, prefix + ".norm3.bias"));
    FloatTensor ff0 = linear_3d(norm3, require_tensor(model, prefix + ".ff.net.0.proj.weight"), &require_tensor(model, prefix + ".ff.net.0.proj.bias"));
    FloatTensor ff1 = geglu(ff0);
    FloatTensor ff2 = linear_3d(ff1, require_tensor(model, prefix + ".ff.net.2.weight"), &require_tensor(model, prefix + ".ff.net.2.bias"));
    add_in_place(hidden, ff2);
    return hidden;
}

[[nodiscard]] FloatTensor transformer2d(const SDXLModel& model,
                                        const std::string& prefix,
                                        const FloatTensor& input,
                                        const FloatTensor& encoder_hidden_states,
                                        std::size_t depth,
                                        std::size_t heads,
                                        std::size_t groups) {
    FloatTensor residual = input;
    FloatTensor hidden = group_norm_2d(input,
                                       require_tensor(model, prefix + ".norm.weight"),
                                       require_tensor(model, prefix + ".norm.bias"),
                                       groups,
                                       1.0e-6F);
    const std::size_t height = input.shape[2];
    const std::size_t width = input.shape[3];
    hidden = flatten_spatial(hidden);
    hidden = linear_3d(hidden, require_tensor(model, prefix + ".proj_in.weight"), &require_tensor(model, prefix + ".proj_in.bias"));
    for (std::size_t i = 0; i < depth; ++i) hidden = transformer_block(model, prefix + ".transformer_blocks." + std::to_string(i), hidden, encoder_hidden_states, heads);
    hidden = linear_3d(hidden, require_tensor(model, prefix + ".proj_out.weight"), &require_tensor(model, prefix + ".proj_out.bias"));
    hidden = unflatten_spatial(hidden, height, width);
    add_in_place(hidden, residual);
    return hidden;
}

[[nodiscard]] FloatTensor add_temb_to_spatial(const FloatTensor& input, const FloatTensor& temb) {
    if (input.shape.size() != 4 || temb.shape.size() != 2 || input.shape[0] != temb.shape[0] || input.shape[1] != temb.shape[1]) throw Error("temb spatial add shape mismatch");
    FloatTensor out = input;
    for (std::size_t b = 0; b < out.shape[0]; ++b) {
        for (std::size_t c = 0; c < out.shape[1]; ++c) {
            const float bias = temb.values[offset2(temb, b, c)];
            for (std::size_t y = 0; y < out.shape[2]; ++y) {
                for (std::size_t x = 0; x < out.shape[3]; ++x) out.values[offset4(out, b, c, y, x)] += bias;
            }
        }
    }
    return out;
}

[[nodiscard]] FloatTensor resnet_block(const SDXLModel& model,
                                       const std::string& prefix,
                                       const FloatTensor& input,
                                       const FloatTensor& temb,
                                       std::size_t groups) {
    FloatTensor hidden = group_norm_2d(input, require_tensor(model, prefix + ".norm1.weight"), require_tensor(model, prefix + ".norm1.bias"), groups);
    silu_in_place(hidden);
    hidden = conv2d(hidden, require_tensor(model, prefix + ".conv1.weight"), &require_tensor(model, prefix + ".conv1.bias"), 1, 1);
    FloatTensor temb_act = temb;
    silu_in_place(temb_act);
    FloatTensor temb_proj = linear_2d(temb_act, require_tensor(model, prefix + ".time_emb_proj.weight"), &require_tensor(model, prefix + ".time_emb_proj.bias"));
    hidden = add_temb_to_spatial(hidden, temb_proj);
    hidden = group_norm_2d(hidden, require_tensor(model, prefix + ".norm2.weight"), require_tensor(model, prefix + ".norm2.bias"), groups);
    silu_in_place(hidden);
    hidden = conv2d(hidden, require_tensor(model, prefix + ".conv2.weight"), &require_tensor(model, prefix + ".conv2.bias"), 1, 1);
    FloatTensor residual = input;
    const ParameterSlot* shortcut = model.graph().find_parameter(prefix + ".conv_shortcut.weight");
    if (shortcut != nullptr && shortcut->tensor.has_value()) {
        residual = conv2d(input, require_tensor(model, prefix + ".conv_shortcut.weight"), &require_tensor(model, prefix + ".conv_shortcut.bias"), 1, 0);
    }
    add_in_place(hidden, residual);
    return hidden;
}

[[nodiscard]] FloatTensor timestep_embedding(const std::vector<float>& timesteps, std::size_t dimension) {
    FloatTensor out = zeros({timesteps.size(), dimension});
    const std::size_t half = dimension / 2;
    for (std::size_t b = 0; b < timesteps.size(); ++b) {
        for (std::size_t i = 0; i < half; ++i) {
            const float exponent = -std::log(10000.0F) * static_cast<float>(i) /
                                   static_cast<float>(std::max<std::size_t>(1, half));
            const float value = timesteps[b] * std::exp(exponent);
            out.values[offset2(out, b, i)] = std::cos(value);
            out.values[offset2(out, b, i + half)] = std::sin(value);
        }
    }
    return out;
}

[[nodiscard]] FloatTensor build_time_embedding(const SDXLModel& model, const SDXLUNetInput& input) {
    const std::size_t batch = input.latents.shape[0];
    std::vector<float> t(batch, input.timestep);
    FloatTensor time_proj = timestep_embedding(t, static_cast<std::size_t>(model.config().unet_channels[0]));
    FloatTensor temb = linear_2d(time_proj, require_tensor(model, "unet.time_embedding.linear_1.weight"), &require_tensor(model, "unet.time_embedding.linear_1.bias"));
    silu_in_place(temb);
    temb = linear_2d(temb, require_tensor(model, "unet.time_embedding.linear_2.weight"), &require_tensor(model, "unet.time_embedding.linear_2.bias"));

    const std::size_t add_dim = static_cast<std::size_t>(model.config().unet_addition_time_embed_dim);
    FloatTensor add_time = zeros({batch, 6 * add_dim});
    for (std::size_t field = 0; field < 6; ++field) {
        std::vector<float> scalar(batch);
        for (std::size_t b = 0; b < batch; ++b) scalar[b] = input.add_time_ids.values[offset2(input.add_time_ids, b, field)];
        FloatTensor field_embed = timestep_embedding(scalar, add_dim);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < add_dim; ++i) add_time.values[offset2(add_time, b, field * add_dim + i)] = field_embed.values[offset2(field_embed, b, i)];
        }
    }
    FloatTensor add_input = zeros({batch, input.pooled_text_embeds.shape[1] + 6 * add_dim});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < input.pooled_text_embeds.shape[1]; ++i) add_input.values[offset2(add_input, b, i)] = input.pooled_text_embeds.values[offset2(input.pooled_text_embeds, b, i)];
        for (std::size_t i = 0; i < 6 * add_dim; ++i) add_input.values[offset2(add_input, b, input.pooled_text_embeds.shape[1] + i)] = add_time.values[offset2(add_time, b, i)];
    }
    FloatTensor add_emb = linear_2d(add_input, require_tensor(model, "unet.add_embedding.linear_1.weight"), &require_tensor(model, "unet.add_embedding.linear_1.bias"));
    silu_in_place(add_emb);
    add_emb = linear_2d(add_emb, require_tensor(model, "unet.add_embedding.linear_2.weight"), &require_tensor(model, "unet.add_embedding.linear_2.bias"));
    add_in_place(temb, add_emb);
    return temb;
}

} // namespace

FloatTensor create_float_tensor(std::vector<std::size_t> shape, std::vector<float> values) {
    return make_tensor(std::move(shape), std::move(values));
}

FloatTensor zeros(std::vector<std::size_t> shape) {
    const std::size_t count = product(shape);
    return make_tensor(std::move(shape), std::vector<float>(count, 0.0F));
}

FloatTensor random_normal(std::vector<std::size_t> shape, std::uint32_t seed, float scale) {
    const std::size_t count = product(shape);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> values(count);
    for (float& value : values) value = dist(rng);
    return make_tensor(std::move(shape), std::move(values));
}

void write_npy_file(const FloatTensor& tensor, const std::string& path) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw Error("failed to open output file: " + path);
    const char magic[] = "\x93NUMPY";
    stream.write(magic, 6);
    stream.put(static_cast<char>(1));
    stream.put(static_cast<char>(0));
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) { if (i) header << ", "; header << tensor.shape[i]; }
    if (tensor.shape.size() == 1) header << ',';
    header << "), }";
    std::string header_text = header.str();
    std::size_t padded_len = header_text.size() + 1;
    while ((10 + padded_len) % 16 != 0) ++padded_len;
    header_text.append(padded_len - header_text.size() - 1, ' ');
    header_text.push_back('\n');
    const std::uint16_t header_len = static_cast<std::uint16_t>(header_text.size());
    stream.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    stream.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    stream.write(reinterpret_cast<const char*>(tensor.values.data()), static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
}

FloatTensor SDXLMicroConditioning::time_ids(std::size_t batch_size) const {
    if (batch_size == 0) throw Error("micro-conditioning batch size cannot be zero");
    if (original_height == 0 || original_width == 0 ||
        target_height == 0 || target_width == 0) {
        throw Error("micro-conditioning original and target dimensions must be nonzero");
    }
    const std::array<float, 6> fields{
        static_cast<float>(original_height),
        static_cast<float>(original_width),
        static_cast<float>(crop_top),
        static_cast<float>(crop_left),
        static_cast<float>(target_height),
        static_cast<float>(target_width)};
    FloatTensor result = zeros({batch_size, fields.size()});
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        std::copy(fields.begin(), fields.end(),
                  result.values.begin() + static_cast<std::ptrdiff_t>(batch * fields.size()));
    }
    return result;
}

SDXLUNet::SDXLUNet(const SDXLModel& model, UNetExecutionOptions options)
    : model_(&model), options_(std::move(options)) {}

FloatTensor SDXLUNet::forward(const FloatTensor& latent_sample,
                              float timestep,
                              const FloatTensor& encoder_hidden_states,
                              const FloatTensor& pooled_text_embeds,
                              const FloatTensor& time_ids) const {
    if (model_ == nullptr) throw Error("SDXLUNet has no model");
    const SDXLUNetInput input{latent_sample, encoder_hidden_states, pooled_text_embeds, time_ids, timestep};
    if (input.latents.shape.size() != 4 || input.latents.shape[1] != 4) throw Error("UNet latents must have shape [B,4,H,W]");
    if (input.latents.values.size() != input.latents.shape[0] * input.latents.shape[1] *
                                          input.latents.shape[2] * input.latents.shape[3]) {
        throw Error("UNet latent storage does not match its shape");
    }
    if (input.encoder_hidden_states.shape.size() != 3 ||
        input.encoder_hidden_states.shape[2] != model_->config().unet_cross_attention_dim) {
        throw Error("UNet encoder_hidden_states cross-attention width is invalid");
    }
    const std::size_t batch = input.latents.shape[0];
    if (input.encoder_hidden_states.shape[0] != batch ||
        input.pooled_text_embeds.shape.size() != 2 || input.pooled_text_embeds.shape[0] != batch ||
        input.add_time_ids.shape != std::vector<std::size_t>{batch, 6}) {
        throw Error("UNet conditioning batch or shape mismatch");
    }
    if (input.encoder_hidden_states.values.size() !=
            input.encoder_hidden_states.shape[0] * input.encoder_hidden_states.shape[1] *
                input.encoder_hidden_states.shape[2] ||
        input.pooled_text_embeds.values.size() !=
            input.pooled_text_embeds.shape[0] * input.pooled_text_embeds.shape[1] ||
        input.add_time_ids.values.size() != batch * 6) {
        throw Error("UNet conditioning storage does not match its shape");
    }
    const std::size_t expected_pooled_width = static_cast<std::size_t>(
        model_->config().unet_addition_input_dim -
        6 * model_->config().unet_addition_time_embed_dim);
    if (input.pooled_text_embeds.shape[1] != expected_pooled_width) {
        throw Error("UNet pooled text embedding width is invalid");
    }
    if (model_->config().unet_channels.size() != model_->config().unet_heads.size() ||
        model_->config().unet_channels.size() != model_->config().unet_transformer_depth.size()) {
        throw Error("UNet configuration vectors have inconsistent lengths");
    }

    FloatTensor temb = build_time_embedding(*model_, input);
    if (options_.progress) options_.progress("conv_in", 0, 0, 1);
    FloatTensor hidden = conv2d(input.latents, require_tensor(*model_, "unet.conv_in.weight"), &require_tensor(*model_, "unet.conv_in.bias"), 1, 1);

    std::vector<FloatTensor> residuals;
    residuals.reserve(16);
    residuals.push_back(hidden);

    const std::size_t groups = static_cast<std::size_t>(model_->config().unet_norm_groups);
    const std::size_t down_blocks = model_->config().unet_channels.size();
    const std::size_t layers_per_block = static_cast<std::size_t>(model_->config().unet_layers_per_block);

    for (std::size_t block = 0; block < down_blocks; ++block) {
        const bool with_attention = block > 0;
        const std::size_t depth = static_cast<std::size_t>(model_->config().unet_transformer_depth[block]);
        const std::size_t heads = static_cast<std::size_t>(model_->config().unet_heads[block]);
        for (std::size_t layer = 0; layer < layers_per_block; ++layer) {
            if (options_.progress) options_.progress("down", block, layer, layers_per_block);
            hidden = resnet_block(*model_, "unet.down_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer), hidden, temb, groups);
            if (with_attention) hidden = transformer2d(*model_, "unet.down_blocks." + std::to_string(block) + ".attentions." + std::to_string(layer), hidden, input.encoder_hidden_states, depth, heads, groups);
            residuals.push_back(hidden);
        }
        if (block + 1 < down_blocks) {
            hidden = conv2d(hidden, require_tensor(*model_, "unet.down_blocks." + std::to_string(block) + ".downsamplers.0.conv.weight"), &require_tensor(*model_, "unet.down_blocks." + std::to_string(block) + ".downsamplers.0.conv.bias"), 2, 1);
            residuals.push_back(hidden);
        }
    }

    if (options_.progress) options_.progress("mid", 0, 0, 3);
    hidden = resnet_block(*model_, "unet.mid_block.resnets.0", hidden, temb, groups);
    if (options_.progress) options_.progress("mid", 0, 1, 3);
    hidden = transformer2d(*model_, "unet.mid_block.attentions.0", hidden, input.encoder_hidden_states,
                           static_cast<std::size_t>(model_->config().unet_transformer_depth.back()),
                           static_cast<std::size_t>(model_->config().unet_heads.back()),
                           groups);
    if (options_.progress) options_.progress("mid", 0, 2, 3);
    hidden = resnet_block(*model_, "unet.mid_block.resnets.1", hidden, temb, groups);

    const std::size_t up_blocks = model_->config().unet_channels.size();
    for (std::size_t block = 0; block < up_blocks; ++block) {
        const std::size_t down_index = up_blocks - 1 - block;
        const std::size_t depth = static_cast<std::size_t>(model_->config().unet_transformer_depth[down_index]);
        const std::size_t heads = static_cast<std::size_t>(model_->config().unet_heads[down_index]);
        for (std::size_t layer = 0; layer < layers_per_block + 1; ++layer) {
            if (residuals.empty()) throw Error("UNet residual stack underflow");
            FloatTensor skip = std::move(residuals.back());
            residuals.pop_back();
            hidden = concat_channels(hidden, skip);
            if (options_.progress) options_.progress("up", block, layer, layers_per_block + 1);
            hidden = resnet_block(*model_, "unet.up_blocks." + std::to_string(block) + ".resnets." + std::to_string(layer), hidden, temb, groups);
            if (down_index > 0) hidden = transformer2d(*model_, "unet.up_blocks." + std::to_string(block) + ".attentions." + std::to_string(layer), hidden, input.encoder_hidden_states, depth, heads, groups);
        }
        if (block + 1 < up_blocks) {
            if (residuals.empty()) throw Error("UNet cannot determine the next upsample target");
            hidden = nearest_upsample(hidden, residuals.back().shape[2], residuals.back().shape[3]);
            hidden = conv2d(hidden, require_tensor(*model_, "unet.up_blocks." + std::to_string(block) + ".upsamplers.0.conv.weight"), &require_tensor(*model_, "unet.up_blocks." + std::to_string(block) + ".upsamplers.0.conv.bias"), 1, 1);
        }
    }
    if (!residuals.empty()) throw Error("UNet left unused skip tensors");

    if (options_.progress) options_.progress("conv_out", 0, 0, 1);
    hidden = group_norm_2d(hidden, require_tensor(*model_, "unet.conv_norm_out.weight"), require_tensor(*model_, "unet.conv_norm_out.bias"), groups);
    silu_in_place(hidden);
    hidden = conv2d(hidden, require_tensor(*model_, "unet.conv_out.weight"), &require_tensor(*model_, "unet.conv_out.bias"), 1, 1);
    if (hidden.shape != input.latents.shape) throw Error("UNet output shape does not match the latent input");
    if (options_.check_finite_outputs) check_finite(hidden, "UNet output");
    return hidden;
}

} // namespace sdxl
