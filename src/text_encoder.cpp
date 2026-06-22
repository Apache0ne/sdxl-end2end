#include "sdxl/text_encoder.hpp"

#include "sdxl/model.hpp"
#include "sdxl/safetensors.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sdxl {
namespace {

[[nodiscard]] std::size_t resolved_thread_count(std::size_t requested) noexcept {
    if (requested != 0) return requested;
    const unsigned detected = std::thread::hardware_concurrency();
    return detected == 0 ? 1 : static_cast<std::size_t>(detected);
}

template <typename Function>
void parallel_tasks(std::size_t task_count, std::size_t thread_count, Function&& function) {
    if (task_count == 0) return;
    const std::size_t workers = std::min(task_count, std::max<std::size_t>(1, thread_count));
    if (workers == 1) {
        for (std::size_t task = 0; task < task_count; ++task) function(task);
        return;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr exception;
    std::mutex exception_mutex;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t task = next.fetch_add(1, std::memory_order_relaxed);
                    if (task >= task_count) break;
                    function(task);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard lock(exception_mutex);
                if (!exception) exception = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (exception) std::rethrow_exception(exception);
}

[[nodiscard]] float half_to_float(std::uint16_t bits) noexcept {
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

[[nodiscard]] float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] float fp8_to_float(std::uint8_t bits, unsigned exponent_bits,
                                 unsigned mantissa_bits, int exponent_bias,
                                 bool finite_only) noexcept {
    const bool negative = (bits & 0x80U) != 0;
    const unsigned exponent_mask = (1U << exponent_bits) - 1U;
    const unsigned mantissa_mask = (1U << mantissa_bits) - 1U;
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    const unsigned mantissa = bits & mantissa_mask;
    float value = 0.0F;
    if (exponent == 0) {
        if (mantissa != 0) {
            value = std::ldexp(static_cast<float>(mantissa) / static_cast<float>(1U << mantissa_bits),
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

class TensorReader final {
public:
    explicit TensorReader(const TensorView& view) : view_(view) {
        if (view_.data == nullptr) throw Error("bound tensor has a null data pointer: " + view_.source_key);
        if (view_.shape.size() != view_.strides_bytes.size()) {
            throw Error("bound tensor has an invalid stride rank: " + view_.source_key);
        }
        switch (view_.dtype) {
        case DType::F16:
        case DType::BF16:
        case DType::F32:
        case DType::F64:
        case DType::F8E4M3FN:
        case DType::F8E5M2:
            break;
        default:
            throw Error("text encoder requires floating-point weights, but " + view_.source_key +
                        " is " + std::string(dtype_name(view_.dtype)));
        }
    }

    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return view_.shape; }

    [[nodiscard]] float at1(std::size_t index) const {
        if (view_.shape.size() != 1 || index >= view_.shape[0]) throw Error("tensor vector access is out of range");
        return read(view_.data + static_cast<std::ptrdiff_t>(index) * view_.strides_bytes[0]);
    }

    [[nodiscard]] float at2(std::size_t row, std::size_t column) const {
        if (view_.shape.size() != 2 || row >= view_.shape[0] || column >= view_.shape[1]) {
            throw Error("tensor matrix access is out of range");
        }
        const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(row) * view_.strides_bytes[0] +
                                      static_cast<std::ptrdiff_t>(column) * view_.strides_bytes[1];
        return read(view_.data + offset);
    }

    void copy_matrix_row(std::size_t row, float* destination, std::size_t columns) const {
        if (view_.shape.size() != 2 || row >= view_.shape[0] || columns > view_.shape[1]) {
            throw Error("tensor row copy is out of range");
        }
        const std::byte* source = view_.data + static_cast<std::ptrdiff_t>(row) * view_.strides_bytes[0];
        const std::int64_t column_stride = view_.strides_bytes[1];
        if (column_stride == static_cast<std::int64_t>(dtype_size(view_.dtype))) {
            switch (view_.dtype) {
            case DType::F32:
                std::memcpy(destination, source, columns * sizeof(float));
                return;
            case DType::F16: {
                for (std::size_t column = 0; column < columns; ++column) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, source + static_cast<std::ptrdiff_t>(column * 2), sizeof(bits));
                    destination[column] = half_to_float(bits);
                }
                return;
            }
            case DType::BF16: {
                for (std::size_t column = 0; column < columns; ++column) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, source + static_cast<std::ptrdiff_t>(column * 2), sizeof(bits));
                    destination[column] = bf16_to_float(bits);
                }
                return;
            }
            default:
                break;
            }
        }
        for (std::size_t column = 0; column < columns; ++column) {
            destination[column] = read(source + static_cast<std::ptrdiff_t>(column) * column_stride);
        }
    }

private:
    [[nodiscard]] float read(const std::byte* address) const noexcept {
        switch (view_.dtype) {
        case DType::F16: {
            std::uint16_t bits = 0;
            std::memcpy(&bits, address, sizeof(bits));
            return half_to_float(bits);
        }
        case DType::BF16: {
            std::uint16_t bits = 0;
            std::memcpy(&bits, address, sizeof(bits));
            return bf16_to_float(bits);
        }
        case DType::F32: {
            float value = 0.0F;
            std::memcpy(&value, address, sizeof(value));
            return value;
        }
        case DType::F64: {
            double value = 0.0;
            std::memcpy(&value, address, sizeof(value));
            return static_cast<float>(value);
        }
        case DType::F8E4M3FN:
            return fp8_to_float(std::to_integer<std::uint8_t>(*address), 4, 3, 7, true);
        case DType::F8E5M2:
            return fp8_to_float(std::to_integer<std::uint8_t>(*address), 5, 2, 15, false);
        default:
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    const TensorView& view_;
};

[[nodiscard]] const TensorView& require_tensor(const SDXLModel& model, const std::string& logical_name) {
    const ParameterSlot* slot = model.graph().find_parameter(logical_name);
    if (slot == nullptr) throw Error("text encoder parameter is not declared: " + logical_name);
    if (!slot->tensor.has_value()) throw Error("text encoder parameter is not loaded: " + logical_name);
    if (slot->tensor->shape != slot->expected_shape) {
        throw Error("text encoder parameter has wrong shape: " + logical_name + " expected " +
                    shape_to_string(slot->expected_shape) + " loaded " + shape_to_string(slot->tensor->shape));
    }
    return *slot->tensor;
}


[[nodiscard]] float dot_product(const float* first, const float* second, std::size_t count) noexcept {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m256 a = _mm256_loadu_ps(first + index);
        const __m256 b = _mm256_loadu_ps(second + index);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; index < count; ++index) result += first[index] * second[index];
    return result;
#else
    float result = 0.0F;
    for (std::size_t index = 0; index < count; ++index) result += first[index] * second[index];
    return result;
#endif
}

[[nodiscard]] std::vector<float> linear(const std::vector<float>& input,
                                        std::size_t rows,
                                        std::size_t input_features,
                                        const TensorView& weight_view,
                                        const TensorView* bias_view,
                                        const TextEncoderExecutionOptions& options) {
    const TensorReader weight(weight_view);
    if (weight.shape().size() != 2 || weight.shape()[1] != input_features) {
        throw Error("linear weight shape does not match input features: " + weight_view.source_key);
    }
    if (input.size() != rows * input_features) throw Error("linear input size is inconsistent");
    const std::size_t output_features = static_cast<std::size_t>(weight.shape()[0]);
    std::unique_ptr<TensorReader> bias;
    if (bias_view != nullptr) {
        bias = std::make_unique<TensorReader>(*bias_view);
        if (bias->shape().size() != 1 || bias->shape()[0] != output_features) {
            throw Error("linear bias shape is invalid: " + bias_view->source_key);
        }
    }

    std::vector<float> output(rows * output_features, 0.0F);
    const std::size_t block = std::max<std::size_t>(1, options.linear_output_block);
    const std::size_t task_count = (output_features + block - 1) / block;
    parallel_tasks(task_count, resolved_thread_count(options.thread_count), [&](std::size_t task) {
        const std::size_t output_begin = task * block;
        const std::size_t output_end = std::min(output_features, output_begin + block);
        const std::size_t block_size = output_end - output_begin;
        std::vector<float> packed(block_size * input_features);
        for (std::size_t local = 0; local < block_size; ++local) {
            weight.copy_matrix_row(output_begin + local,
                                   packed.data() + local * input_features,
                                   input_features);
        }
        for (std::size_t local = 0; local < block_size; ++local) {
            const std::size_t output_column = output_begin + local;
            const float* packed_row = packed.data() + local * input_features;
            const float bias_value = bias ? bias->at1(output_column) : 0.0F;
            for (std::size_t row = 0; row < rows; ++row) {
                output[row * output_features + output_column] =
                    dot_product(input.data() + row * input_features, packed_row, input_features) + bias_value;
            }
        }
    });
    return output;
}

[[nodiscard]] std::vector<float> layer_norm(const std::vector<float>& input,
                                            std::size_t rows,
                                            std::size_t width,
                                            const TensorView& weight_view,
                                            const TensorView& bias_view,
                                            float epsilon,
                                            std::size_t thread_count) {
    if (input.size() != rows * width) throw Error("layer norm input size is inconsistent");
    const TensorReader weight(weight_view);
    const TensorReader bias(bias_view);
    if (weight.shape() != std::vector<std::uint64_t>{width} ||
        bias.shape() != std::vector<std::uint64_t>{width}) {
        throw Error("layer norm parameter shape is invalid");
    }
    std::vector<float> output(input.size());
    parallel_tasks(rows, resolved_thread_count(thread_count), [&](std::size_t row) {
        const float* source = input.data() + row * width;
        float mean = 0.0F;
        for (std::size_t column = 0; column < width; ++column) mean += source[column];
        mean /= static_cast<float>(width);
        float variance = 0.0F;
        for (std::size_t column = 0; column < width; ++column) {
            const float centered = source[column] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(width);
        const float inv_std = 1.0F / std::sqrt(variance + epsilon);
        float* destination = output.data() + row * width;
        for (std::size_t column = 0; column < width; ++column) {
            destination[column] = (source[column] - mean) * inv_std * weight.at1(column) + bias.at1(column);
        }
    });
    return output;
}

void add_in_place(std::vector<float>& destination, const std::vector<float>& source,
                  std::size_t thread_count) {
    if (destination.size() != source.size()) throw Error("residual tensor sizes do not match");
    const std::size_t chunk = 16'384;
    const std::size_t tasks = (destination.size() + chunk - 1) / chunk;
    parallel_tasks(tasks, resolved_thread_count(thread_count), [&](std::size_t task) {
        const std::size_t begin = task * chunk;
        const std::size_t end = std::min(destination.size(), begin + chunk);
        for (std::size_t index = begin; index < end; ++index) destination[index] += source[index];
    });
}

void activate_in_place(std::vector<float>& values, std::string_view activation,
                       std::size_t thread_count) {
    const std::size_t chunk = 16'384;
    const std::size_t tasks = (values.size() + chunk - 1) / chunk;
    parallel_tasks(tasks, resolved_thread_count(thread_count), [&](std::size_t task) {
        const std::size_t begin = task * chunk;
        const std::size_t end = std::min(values.size(), begin + chunk);
        if (activation == "quick_gelu") {
            for (std::size_t index = begin; index < end; ++index) {
                const float x = values[index];
                values[index] = x / (1.0F + std::exp(-1.702F * x));
            }
        } else if (activation == "gelu") {
            constexpr float inverse_sqrt_two = 0.7071067811865475244F;
            for (std::size_t index = begin; index < end; ++index) {
                const float x = values[index];
                values[index] = 0.5F * x * (1.0F + std::erf(x * inverse_sqrt_two));
            }
        } else {
            throw Error("unsupported CLIP activation: " + std::string(activation));
        }
    });
}

[[nodiscard]] std::vector<float> causal_self_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const TokenizedBatch& tokens,
    std::size_t hidden_size,
    std::size_t heads,
    bool use_attention_mask,
    std::size_t thread_count) {
    const std::size_t batch_size = tokens.batch_size;
    const std::size_t sequence = tokens.sequence_length;
    if (heads == 0 || hidden_size % heads != 0) throw Error("CLIP attention head configuration is invalid");
    const std::size_t head_size = hidden_size / heads;
    const std::size_t expected = batch_size * sequence * hidden_size;
    if (query.size() != expected || key.size() != expected || value.size() != expected) {
        throw Error("CLIP attention tensor size is invalid");
    }
    std::vector<float> output(expected, 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    const std::size_t tasks = batch_size * heads;
    parallel_tasks(tasks, resolved_thread_count(thread_count), [&](std::size_t task) {
        const std::size_t batch = task / heads;
        const std::size_t head = task % heads;
        const std::size_t channel_offset = head * head_size;
        std::vector<float> scores(sequence, -std::numeric_limits<float>::infinity());
        std::vector<float> probabilities(sequence, 0.0F);
        for (std::size_t query_position = 0; query_position < sequence; ++query_position) {
            const float* q = query.data() + (batch * sequence + query_position) * hidden_size + channel_offset;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t key_position = 0; key_position <= query_position; ++key_position) {
                if (use_attention_mask && tokens.mask(batch)[key_position] == 0) {
                    scores[key_position] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* k = key.data() + (batch * sequence + key_position) * hidden_size + channel_offset;
                const float score = dot_product(q, k, head_size) * scale;
                scores[key_position] = score;
                maximum = std::max(maximum, score);
            }
            if (!std::isfinite(maximum)) throw Error("CLIP attention row has no visible keys");
            float denominator = 0.0F;
            for (std::size_t key_position = 0; key_position <= query_position; ++key_position) {
                const float probability = std::isfinite(scores[key_position])
                                              ? std::exp(scores[key_position] - maximum)
                                              : 0.0F;
                probabilities[key_position] = probability;
                denominator += probability;
            }
            const float inverse_denominator = 1.0F / denominator;
            for (std::size_t key_position = 0; key_position <= query_position; ++key_position) {
                probabilities[key_position] *= inverse_denominator;
            }

            float* destination = output.data() +
                                 (batch * sequence + query_position) * hidden_size + channel_offset;
            for (std::size_t key_position = 0; key_position <= query_position; ++key_position) {
                const float probability = probabilities[key_position];
                const float* v = value.data() +
                                 (batch * sequence + key_position) * hidden_size + channel_offset;
                for (std::size_t channel = 0; channel < head_size; ++channel) {
                    destination[channel] += probability * v[channel];
                }
            }
        }
    });
    return output;
}

[[nodiscard]] std::vector<float> embed_tokens(const TokenizedBatch& tokens,
                                               const TensorView& token_embedding_view,
                                               const TensorView& position_embedding_view,
                                               std::size_t hidden_size,
                                               std::size_t thread_count) {
    const TensorReader token_embedding(token_embedding_view);
    const TensorReader position_embedding(position_embedding_view);
    if (token_embedding.shape().size() != 2 || token_embedding.shape()[1] != hidden_size) {
        throw Error("CLIP token embedding shape is invalid");
    }
    if (position_embedding.shape().size() != 2 || position_embedding.shape()[1] != hidden_size ||
        position_embedding.shape()[0] < tokens.sequence_length) {
        throw Error("CLIP position embedding shape is invalid");
    }
    std::vector<float> output(tokens.batch_size * tokens.sequence_length * hidden_size);
    const std::size_t rows = tokens.batch_size * tokens.sequence_length;
    parallel_tasks(rows, resolved_thread_count(thread_count), [&](std::size_t row) {
        const std::size_t batch = row / tokens.sequence_length;
        const std::size_t position = row % tokens.sequence_length;
        const std::int32_t token_id = tokens.ids(batch)[position];
        if (token_id < 0 || static_cast<std::uint64_t>(token_id) >= token_embedding.shape()[0]) {
            throw Error("CLIP token ID is outside the embedding vocabulary");
        }
        float* destination = output.data() + row * hidden_size;
        for (std::size_t channel = 0; channel < hidden_size; ++channel) {
            destination[channel] = token_embedding.at2(static_cast<std::size_t>(token_id), channel) +
                                   position_embedding.at2(position, channel);
        }
    });
    return output;
}

void check_finite(const std::vector<float>& values, std::string_view name) {
    for (const float value : values) {
        if (!std::isfinite(value)) throw Error(std::string(name) + " contains NaN or infinity");
    }
}

[[nodiscard]] FloatTensor make_tensor(std::vector<std::size_t> shape, std::vector<float> values) {
    std::size_t expected = 1;
    for (const std::size_t dimension : shape) expected *= dimension;
    if (expected != values.size()) throw Error("FloatTensor shape does not match its storage");
    return FloatTensor{std::move(shape), std::move(values)};
}

} // namespace

CLIPTextEncoder::CLIPTextEncoder(const SDXLModel& model,
                                 std::string component_name,
                                 SDXLConfig::TextEncoder config,
                                 TextEncoderExecutionOptions options)
    : model_(&model),
      component_name_(std::move(component_name)),
      config_(std::move(config)),
      options_(std::move(options)) {
    if (config_.hidden_size == 0 || config_.layers == 0 || config_.heads == 0 ||
        config_.hidden_size % config_.heads != 0) {
        throw Error("invalid CLIP text encoder configuration for " + component_name_);
    }
}

CLIPTextEncoderOutput CLIPTextEncoder::forward(const TokenizedBatch& tokens) const {
    if (model_ == nullptr) throw Error("CLIP text encoder has no model");
    if (tokens.batch_size == 0 || tokens.sequence_length == 0) throw Error("CLIP token batch is empty");
    if (tokens.sequence_length > config_.max_positions) throw Error("CLIP token sequence exceeds position embeddings");
    if (tokens.input_ids.size() != tokens.batch_size * tokens.sequence_length ||
        tokens.attention_mask.size() != tokens.batch_size * tokens.sequence_length) {
        throw Error("CLIP token batch storage is inconsistent");
    }

    const std::size_t batch = tokens.batch_size;
    const std::size_t sequence = tokens.sequence_length;
    const std::size_t hidden_size = static_cast<std::size_t>(config_.hidden_size);
    const std::size_t intermediate_size = static_cast<std::size_t>(config_.intermediate_size);
    const std::size_t rows = batch * sequence;
    const std::string prefix = component_name_ + ".text_model.";

    std::vector<float> hidden = embed_tokens(
        tokens,
        require_tensor(*model_, prefix + "embeddings.token_embedding.weight"),
        require_tensor(*model_, prefix + "embeddings.position_embedding.weight"),
        hidden_size,
        options_.thread_count);

    std::vector<float> penultimate;
    if (config_.layers == 1) penultimate = hidden;

    for (std::size_t layer = 0; layer < static_cast<std::size_t>(config_.layers); ++layer) {
        if (options_.progress) options_.progress(component_name_, layer, static_cast<std::size_t>(config_.layers));
        const std::string layer_prefix = prefix + "encoder.layers." + std::to_string(layer) + ".";

        std::vector<float> normalized = layer_norm(
            hidden, rows, hidden_size,
            require_tensor(*model_, layer_prefix + "layer_norm1.weight"),
            require_tensor(*model_, layer_prefix + "layer_norm1.bias"),
            1.0e-5F, options_.thread_count);

        const auto project = [&](std::string_view name) {
            const std::string weight = layer_prefix + "self_attn." + std::string(name) + ".weight";
            const std::string bias = layer_prefix + "self_attn." + std::string(name) + ".bias";
            return linear(normalized, rows, hidden_size,
                          require_tensor(*model_, weight),
                          &require_tensor(*model_, bias), options_);
        };
        std::vector<float> query = project("q_proj");
        std::vector<float> key = project("k_proj");
        std::vector<float> value = project("v_proj");
        std::vector<float> attention = causal_self_attention(
            query, key, value, tokens, hidden_size, static_cast<std::size_t>(config_.heads),
            options_.use_attention_mask, options_.thread_count);
        std::vector<float> attention_output = linear(
            attention, rows, hidden_size,
            require_tensor(*model_, layer_prefix + "self_attn.out_proj.weight"),
            &require_tensor(*model_, layer_prefix + "self_attn.out_proj.bias"), options_);
        add_in_place(hidden, attention_output, options_.thread_count);

        normalized = layer_norm(
            hidden, rows, hidden_size,
            require_tensor(*model_, layer_prefix + "layer_norm2.weight"),
            require_tensor(*model_, layer_prefix + "layer_norm2.bias"),
            1.0e-5F, options_.thread_count);
        std::vector<float> intermediate = linear(
            normalized, rows, hidden_size,
            require_tensor(*model_, layer_prefix + "mlp.fc1.weight"),
            &require_tensor(*model_, layer_prefix + "mlp.fc1.bias"), options_);
        activate_in_place(intermediate, config_.activation, options_.thread_count);
        std::vector<float> mlp_output = linear(
            intermediate, rows, intermediate_size,
            require_tensor(*model_, layer_prefix + "mlp.fc2.weight"),
            &require_tensor(*model_, layer_prefix + "mlp.fc2.bias"), options_);
        add_in_place(hidden, mlp_output, options_.thread_count);

        if (layer + 2 == static_cast<std::size_t>(config_.layers)) penultimate = hidden;
    }
    if (options_.progress) {
        options_.progress(component_name_, static_cast<std::size_t>(config_.layers),
                          static_cast<std::size_t>(config_.layers));
    }

    std::vector<float> last_hidden = layer_norm(
        hidden, rows, hidden_size,
        require_tensor(*model_, prefix + "final_layer_norm.weight"),
        require_tensor(*model_, prefix + "final_layer_norm.bias"),
        1.0e-5F, options_.thread_count);

    std::vector<float> pooled(batch * hidden_size);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        const std::int32_t* ids = tokens.ids(batch_index);
        std::size_t pooled_position = 0;
        for (std::size_t position = 1; position < sequence; ++position) {
            if (ids[position] > ids[pooled_position]) pooled_position = position;
        }
        const float* source = last_hidden.data() +
                              (batch_index * sequence + pooled_position) * hidden_size;
        std::copy_n(source, hidden_size, pooled.data() + batch_index * hidden_size);
    }

    std::vector<float> projected;
    if (config_.with_projection) {
        projected = linear(pooled, batch, hidden_size,
                           require_tensor(*model_, component_name_ + ".text_projection.weight"),
                           nullptr, options_);
    }

    if (penultimate.empty()) throw Error("CLIP penultimate hidden state was not captured");
    if (options_.check_finite_outputs) {
        check_finite(last_hidden, component_name_ + " last hidden state");
        check_finite(penultimate, component_name_ + " penultimate hidden state");
        check_finite(pooled, component_name_ + " pooled output");
        if (!projected.empty()) check_finite(projected, component_name_ + " projected text embeddings");
    }

    CLIPTextEncoderOutput output;
    output.last_hidden_state = make_tensor({batch, sequence, hidden_size}, std::move(last_hidden));
    output.penultimate_hidden_state = make_tensor({batch, sequence, hidden_size}, std::move(penultimate));
    output.pooled_output = make_tensor({batch, hidden_size}, std::move(pooled));
    if (!projected.empty()) {
        output.text_embeds = make_tensor(
            {batch, static_cast<std::size_t>(config_.projection_dim)}, std::move(projected));
    }
    return output;
}

SDXLTextConditioner::SDXLTextConditioner(const SDXLModel& model,
                                         CLIPTokenizer tokenizer,
                                         CLIPTokenizer tokenizer_2,
                                         TextEncoderExecutionOptions options)
    : model_(&model),
      tokenizer_(std::move(tokenizer)),
      tokenizer_2_(std::move(tokenizer_2)),
      clip_l_(model, "text_encoder", model.config().clip_l, options),
      openclip_(model, "text_encoder_2", model.config().openclip_big_g, std::move(options)) {}

SDXLTextConditioner SDXLTextConditioner::builtin_sdxl(
    const SDXLModel& model,
    TextEncoderExecutionOptions options) {
    return SDXLTextConditioner(model,
                               CLIPTokenizer::sdxl_clip_l(),
                               CLIPTokenizer::sdxl_openclip_big_g(),
                               std::move(options));
}

SDXLTextConditioner SDXLTextConditioner::from_model_directory(
    const SDXLModel& model,
    const std::filesystem::path& model_directory,
    TextEncoderExecutionOptions options) {
    const std::filesystem::path first_directory = model_directory / "tokenizer";
    const std::filesystem::path second_candidate = model_directory / "tokenizer_2";
    const std::filesystem::path second_directory = std::filesystem::is_directory(second_candidate)
                                                       ? second_candidate
                                                       : first_directory;
    return SDXLTextConditioner(model,
                               CLIPTokenizer(first_directory),
                               CLIPTokenizer(second_directory),
                               std::move(options));
}

SDXLPromptConditioning SDXLTextConditioner::encode(const std::vector<std::string>& prompts) const {
    return encode(prompts, prompts);
}

SDXLPromptConditioning SDXLTextConditioner::encode(
    const std::vector<std::string>& prompts,
    const std::vector<std::string>& prompts_2) const {
    if (model_ == nullptr) throw Error("SDXL text conditioner has no model");
    if (prompts.empty()) throw Error("SDXL prompt batch cannot be empty");
    std::vector<std::string> second_prompts = prompts_2.empty() ? prompts : prompts_2;
    if (second_prompts.size() == 1 && prompts.size() > 1) {
        second_prompts.assign(prompts.size(), second_prompts.front());
    }
    if (second_prompts.size() != prompts.size()) {
        throw Error("prompt_2 batch must have one item or match the prompt batch size");
    }
    TokenizedBatch first_tokens = tokenizer_.encode_batch(prompts);
    TokenizedBatch second_tokens = tokenizer_2_.encode_batch(second_prompts);
    if (first_tokens.batch_size != second_tokens.batch_size ||
        first_tokens.sequence_length != second_tokens.sequence_length) {
        throw Error("SDXL tokenizers produced incompatible batch shapes");
    }

    CLIPTextEncoderOutput first = clip_l_.forward(first_tokens);
    CLIPTextEncoderOutput second = openclip_.forward(second_tokens);
    const std::size_t batch = first_tokens.batch_size;
    const std::size_t sequence = first_tokens.sequence_length;
    const std::size_t first_width = static_cast<std::size_t>(model_->config().clip_l.hidden_size);
    const std::size_t second_width = static_cast<std::size_t>(model_->config().openclip_big_g.hidden_size);
    const std::size_t combined_width = first_width + second_width;

    std::vector<float> combined(batch * sequence * combined_width);
    for (std::size_t row = 0; row < batch * sequence; ++row) {
        float* destination = combined.data() + row * combined_width;
        std::copy_n(first.penultimate_hidden_state.data() + row * first_width,
                    first_width, destination);
        std::copy_n(second.penultimate_hidden_state.data() + row * second_width,
                    second_width, destination + first_width);
    }
    if (second.text_embeds.values.empty()) {
        throw Error("SDXL second text encoder did not produce projected pooled embeddings");
    }

    SDXLPromptConditioning result;
    result.clip_l_tokens = std::move(first_tokens);
    result.openclip_tokens = std::move(second_tokens);
    result.prompt_embeds = make_tensor({batch, sequence, combined_width}, std::move(combined));
    result.pooled_prompt_embeds = std::move(second.text_embeds);
    return result;
}

SDXLPromptConditioning SDXLTextConditioner::encode(std::string_view prompt) const {
    return encode(std::vector<std::string>{std::string(prompt)});
}

SDXLPromptConditioning SDXLTextConditioner::encode(std::string_view prompt,
                                                    std::string_view prompt_2) const {
    return encode(std::vector<std::string>{std::string(prompt)},
                  std::vector<std::string>{std::string(prompt_2)});
}

SDXLClassifierFreeConditioning SDXLTextConditioner::encode_classifier_free(
    const std::vector<std::string>& prompts,
    const std::vector<std::string>& negative_prompts) const {
    if (prompts.empty()) throw Error("positive prompt batch cannot be empty");
    std::vector<std::string> negatives = negative_prompts;
    if (negatives.empty()) negatives.assign(prompts.size(), "");
    if (negatives.size() == 1 && prompts.size() > 1) negatives.assign(prompts.size(), negatives.front());
    if (negatives.size() != prompts.size()) {
        throw Error("negative prompt batch must have one item or match the positive batch size");
    }
    SDXLClassifierFreeConditioning result;
    result.positive = encode(prompts);
    result.negative = encode(negatives);
    return result;
}

} // namespace sdxl
